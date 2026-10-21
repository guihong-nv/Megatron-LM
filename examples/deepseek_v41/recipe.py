# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generate auditable V4.1 text recipes without importing PyTorch or allocating a model."""

import argparse
import copy
import hashlib
import json
import math
import os
import shlex
import sys
import uuid
from pathlib import Path

REFERENCE = Path(__file__).with_name("config_flash.json")
REFERENCE_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json"
# Retain both Engram owners and all attention roles, including encoder and decoder reuse.
PROXY_BLOCK_IDS = (0, 1, 2, 3, 14, 20, 24, 25)
PROFILES = ("full", "depth-proxy", "mechanism-smoke")


def _prime_rows(budget: int, layers: int, orders: int, heads: int) -> list[int]:
    """Match the distinct-prime Engram layout using only the Python standard library."""
    seen = set()
    result = []
    for _ in range(layers):
        rows = 0
        for _ in range(orders):
            current = budget - 1
            for _ in range(heads):
                current += 1
                while (
                    current in seen
                    or current < 2
                    or any(current % divisor == 0 for divisor in range(2, math.isqrt(current) + 1))
                ):
                    current += 1
                seen.add(current)
                rows += current
        result.append(rows)
    return result


def build_profile(profile: str, reference: dict | None = None) -> dict:
    """Select logical blocks and record every deliberate deviation from released geometry."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    original = (reference or json.loads(REFERENCE.read_text(encoding="utf-8")))["text_config"]
    text = copy.deepcopy(original)
    selected = (
        list(range(original["num_hidden_layers"])) if profile == "full" else list(PROXY_BLOCK_IDS)
    )
    mapping = {source: target for target, source in enumerate(selected)}
    text["num_hidden_layers"] = len(selected)
    text["compress_ratios"] = [original["compress_ratios"][i] for i in selected]
    for field in ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids"):
        text[field] = [mapping[i] for i in original[field] if i in mapping]
    text["candidate_source_layer_id"] = mapping[original["candidate_source_layer_id"]]
    if profile == "mechanism-smoke":
        # This is explicitly a reduced-geometry execution test, never the depth-only proxy.
        text.update(
            hidden_size=256,
            q_lora_rank=128,
            o_lora_rank=32,
            moe_intermediate_size=256,
            n_routed_experts=8,
            vocab_size=256,
            engram_vocab_size=101,
            engram_compressed_vocab_size=256,
            candidate_topk_blocks=64,
        )
        text["engram_num_embeddings"] = _prime_rows(
            text["engram_vocab_size"],
            len(text["engram_layer_ids"]),
            text["engram_max_ngram_size"] - 1,
            text["engram_n_heads"],
        )
    roles = []
    for index, ratio in enumerate(text["compress_ratios"]):
        roles.append(
            "SWA"
            if not ratio
            else (
                "Full"
                if index in text["kv_source_layer_ids"]
                else "Reindex" if index in text["index_source_layer_ids"] else "Reuse"
            )
        )
    changes = {
        key: {"released": value, "recipe": text[key]}
        for key, value in original.items()
        if text[key] != value
    }
    engram = {
        key.removeprefix("engram_"): value
        for key, value in text.items()
        if key.startswith("engram_")
    }
    return {
        "profile": profile,
        "source": REFERENCE_URL,
        "source_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        "scope": "BF16 text backbone and Engram, trained from scratch with baseline Adam",
        "excluded": ["vision", "DSpark", "FP8/FP4 QAT", "released optimizer recipe"],
        "original_logical_block_ids": selected,
        "attention_roles": roles,
        "changes_from_released_text_config": changes,
        "text_config": text,
        "engram_config": engram,
    }


def validate_topology(recipe: dict, world_size: int, pp: int, cp: int, ep: int) -> None:
    """Reject invalid static TP1 layouts before starting torchrun."""
    if min(world_size, pp, cp, ep) < 1:
        raise ValueError("World size, PP, CP and EP must be positive")
    if world_size % (pp * cp):
        raise ValueError("World size must be divisible by PP * CP (TP is fixed at one)")
    if world_size % (pp * ep):
        raise ValueError("World size must be divisible by PP * EP")
    text = recipe["text_config"]
    if text["num_hidden_layers"] % pp:
        raise ValueError("This recipe requires equal numbers of complete VE blocks per PP stage")
    if text["n_routed_experts"] % ep:
        raise ValueError("The routed expert count must be divisible by EP")


def memory_estimate(recipe: dict, gpu_memory_gib: float = 180.0) -> dict:
    """Report an ideal aggregate lower bound, excluding dense weights and all activations."""
    text = recipe["text_config"]
    expert = (
        text["num_hidden_layers"]
        * 3
        * text["hidden_size"]
        * text["moe_intermediate_size"]
        * text["n_routed_experts"]
    )
    engram = sum(text["engram_num_embeddings"]) * text["engram_head_dim"]
    parameters = expert + engram
    return {
        "routed_expert_parameters": expert,
        "engram_table_parameters": engram,
        "parameter_lower_bound": parameters,
        "bf16_weights_lower_bound_gib": parameters * 2 / 2**30,
        "adam_state_and_weights_lower_bound_gib": parameters * 18 / 2**30,
        "minimum_gpus_at_75_percent_memory_lower_bound": math.ceil(
            parameters * 18 / (gpu_memory_gib * 2**30 * 0.75)
        ),
        "assumed_gpu_memory_gib": gpu_memory_gib,
        "assumptions": "2-byte parameters + 4-byte gradients + 4-byte master weights + "
        "8-byte Adam moments; ideal sharding, no replicas, dense weights, "
        "activations, communication buffers or uneven stage ownership. "
        "This is a capacity lower bound, not a validated fit guarantee.",
    }


def training_args(recipe: dict, args: argparse.Namespace) -> list[str]:
    """Translate the complete text architecture to the existing pretrain_hybrid entrypoint."""
    text = recipe["text_config"]
    depth = text["num_hidden_layers"]
    pattern = "|".join(["VE" * (depth // args.pp)] * args.pp)
    rope = text["rope_scaling"]
    values = {
        "dsv4-version": "v4.1",
        "hybrid-layer-pattern": pattern,
        "num-layers": 2 * depth,
        "hidden-size": text["hidden_size"],
        "num-attention-heads": text["num_attention_heads"],
        "q-lora-rank": text["q_lora_rank"],
        "v-head-dim": text["head_dim"],
        "qk-pos-emb-head-dim": text["qk_rope_head_dim"],
        "output-projection-groups": text["o_groups"],
        "output-projection-lora-rank": text["o_lora_rank"],
        "normalization": "RMSNorm",
        "norm-epsilon": text["rms_norm_eps"],
        "rotary-base": text["rope_theta"],
        "csa-compress-rotary-base": text["compress_rope_theta"],
        "rotary-scaling-factor": rope["factor"],
        "original-max-position-embeddings": rope["original_max_position_embeddings"],
        "mscale": 0,
        "mscale-all-dim": 0,
        "csa-compress-ratios": json.dumps(
            [v for ratio in text["compress_ratios"] for v in (ratio, 0)], separators=(",", ":")
        ),
        "csa-window-size": text["sliding_window"],
        "csa2-candidate-source-layer": 2 * text["candidate_source_layer_id"],
        "csa2-candidate-topk-blocks": text["candidate_topk_blocks"],
        "csa2-candidate-block-size": text["candidate_block_size"],
        "dsa-indexer-n-heads": text["index_n_heads"],
        "dsa-indexer-head-dim": text["index_head_dim"],
        "dsa-indexer-topk": text["index_topk"],
        "dsa-kernel-backend": "cudnn",
        "dsa-indexer-loss-coeff": args.indexer_loss_coeff,
        "mhc-num-residual-streams": text["hc_mult"],
        "mhc-sinkhorn-iterations": text["hc_sinkhorn_iters"],
        "mhc-epsilon": text["hc_eps"],
        "num-experts": text["n_routed_experts"],
        "moe-router-topk": text["num_experts_per_tok"],
        "moe-ffn-hidden-size": text["moe_intermediate_size"],
        "moe-shared-expert-intermediate-size": text["moe_intermediate_size"]
        * text["n_shared_experts"],
        "moe-router-score-function": text["scoring_func"],
        "moe-router-topk-scaling-factor": text["routed_scaling_factor"],
        "moe-router-dtype": "fp32",
        "moe-router-load-balancing-type": "none",
        "moe-aux-loss-coeff": 0,
        "moe-token-dispatcher-type": "alltoall",
        "activation-func-clamp-value": text["swiglu_limit"],
        "engram-config": str(args.run_dir / "engram.json"),
        "engram-token-map": str(args.engram_token_map or args.run_dir / "token_map.pt"),
        "tensor-model-parallel-size": 1,
        "pipeline-model-parallel-size": args.pp,
        "context-parallel-size": args.cp,
        "expert-model-parallel-size": args.ep,
        "expert-tensor-parallel-size": 1,
        "attention-dropout": 0,
        "hidden-dropout": 0,
        "seq-length": args.seq_length,
        "max-position-embeddings": text["max_position_embeddings"],
        "micro-batch-size": args.micro_batch_size,
        "global-batch-size": args.global_batch_size,
        "train-iters": args.steps,
        "lr": args.lr,
        "min-lr": args.lr,
        "lr-decay-style": "constant",
        "optimizer": "adam",
        "adam-beta1": 0.9,
        "adam-beta2": 0.95,
        "adam-eps": 1e-8,
        "weight-decay": 0.1,
        "clip-grad": 1.0,
        "init-method-std": text["initializer_range"],
        "eval-interval": args.steps,
        "eval-iters": 1,
        "num-workers": 0,
        "log-interval": 1,
        "save": str(args.run_dir / "checkpoints"),
        "load": str(args.run_dir / "checkpoints"),
        "save-interval": args.save_interval,
        "ckpt-format": "torch_dist",
        "vocab-size": text["vocab_size"],
    }
    result = [part for key, value in values.items() for part in ("--" + key, str(value))]
    for field in ("kv", "index"):
        result += [f"--csa2-{field}-source-layers"]
        result += [str(2 * i) for i in text[f"{field}_source_layer_ids"]]
    result += [
        "--" + flag
        for flag in (
            "multi-latent-attention",
            "qk-layernorm",
            "no-dsa-indexer-rotate-activation",
            "dsa-indexer-use-sparse-loss",
            "enable-mhc-connections",
            "mhc-single-pass",
            "use-fused-mhc",
            "moe-router-enable-expert-bias",
            "moe-grouped-gemm",
            "swiglu",
            "disable-bias-linear",
            "bf16",
            "use-distributed-optimizer",
            "use-checkpoint-opt-param-scheduler",
            "untie-embeddings-and-output-weights",
            "no-create-attention-mask-in-dataloader",
        )
    ]
    if args.cp > 1:
        result += [
            "--attention-cp-layout",
            "contiguous",
            "--linear-cp-layout",
            "contiguous",
            "--dataloader-inter-document-masking",
        ]
    if args.full_recompute:
        result += [
            "--recompute-granularity",
            "full",
            "--recompute-method",
            "uniform",
            "--recompute-num-layers",
            "1",
        ]
    if args.data_path:
        result += [
            "--data-path",
            args.data_path,
            "--tokenizer-type",
            "HuggingFaceTokenizer",
            "--tokenizer-model",
            args.tokenizer_model,
        ]
    else:
        result += ["--mock-data", "--tokenizer-type", "NullTokenizer"]
    return result


def parser() -> argparse.ArgumentParser:
    """Create the standalone recipe/launcher command-line parser."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--profile", choices=PROFILES, default="depth-proxy")
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--nnodes", type=int, default=int(os.environ.get("NNODES", "1")))
    result.add_argument(
        "--nproc-per-node", type=int, default=int(os.environ.get("GPUS_PER_NODE", "8"))
    )
    result.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", "0")))
    result.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    result.add_argument(
        "--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29500"))
    )
    for name in ("ep", "pp", "cp"):
        result.add_argument("--" + name, type=int, default=1)
    result.add_argument("--seq-length", type=int, default=1024)
    result.add_argument("--steps", type=int, default=8)
    result.add_argument("--save-interval", type=int, default=4)
    result.add_argument("--micro-batch-size", type=int, default=1)
    result.add_argument("--global-batch-size", type=int, default=8)
    result.add_argument("--indexer-loss-coeff", type=float, default=0.01)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--engram-token-map", type=Path)
    result.add_argument("--tokenizer-model")
    result.add_argument("--data-path")
    result.add_argument("--gpu-memory-gib", type=float, default=180.0)
    result.add_argument("--full-recompute", action="store_true")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--dry-run", action="store_false", dest="execute")
    return result


def _write_json(path: Path, value: dict | list) -> None:
    """Publish complete manifests atomically when several node launchers share a directory."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    """Materialize the recipe, then optionally replace this process with torchrun."""
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    if args.engram_token_map:
        args.engram_token_map = args.engram_token_map.resolve()
    try:
        recipe = build_profile(args.profile)
        world = args.nnodes * args.nproc_per_node
        validate_topology(recipe, world, args.pp, args.cp, args.ep)
        if (
            min(
                args.seq_length,
                args.steps,
                args.micro_batch_size,
                args.global_batch_size,
                args.save_interval,
            )
            < 1
            or args.gpu_memory_gib <= 0
        ):
            raise ValueError(
                "Lengths, steps, batch sizes, save interval and GPU memory must be positive"
            )
        if not 0 <= args.node_rank < args.nnodes or args.nproc_per_node < 1:
            raise ValueError("Invalid node rank or GPUs per node")
        if args.seq_length % (2 * args.cp):
            raise ValueError("Sequence length must be divisible by 2 * CP for ratio-2 compression")
        if args.global_batch_size % (args.micro_batch_size * world // (args.pp * args.cp)):
            raise ValueError("Global batch size must be divisible by micro batch size * DP size")
        if args.data_path and not args.tokenizer_model:
            raise ValueError("--data-path requires --tokenizer-model")
        if args.profile == "mechanism-smoke" and (args.data_path or args.engram_token_map):
            raise ValueError(
                "mechanism-smoke uses only synthetic data and its explicit identity map"
            )
        if (
            args.execute
            and args.profile != "mechanism-smoke"
            and (not args.engram_token_map or not args.engram_token_map.is_file())
        ):
            raise ValueError(
                "Faithful profiles require --engram-token-map from prepare_token_map.py"
            )
    except ValueError as error:
        arg_parser.error(str(error))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    recipe["memory_lower_bound"] = memory_estimate(recipe, args.gpu_memory_gib)
    recipe["data"] = "indexed training data" if args.data_path else "synthetic mock data"
    recipe["parallelism"] = {
        "tp": 1,
        "pp": args.pp,
        "cp": args.cp,
        "ep": args.ep,
        "world_size": world,
    }
    _write_json(args.run_dir / "engram.json", recipe["engram_config"])
    _write_json(args.run_dir / f"recipe-rank{args.node_rank}.json", recipe)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(args.nnodes),
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--node-rank",
        str(args.node_rank),
        "--master-addr",
        args.master_addr,
        "--master-port",
        str(args.master_port),
        "pretrain_hybrid.py",
        *training_args(recipe, args),
    ]
    _write_json(args.run_dir / f"command-rank{args.node_rank}.json", command)
    # This standalone CLI writes its inspection summary and launch command to stdout.
    # pylint: disable=bad-builtin
    print(
        json.dumps(
            {
                "profile": args.profile,
                "original_blocks": recipe["original_logical_block_ids"],
                "attention_roles": recipe["attention_roles"],
                "memory_lower_bound": recipe["memory_lower_bound"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(shlex.join(command), flush=True)
    # pylint: enable=bad-builtin
    if args.execute:
        import torch

        token_path = args.engram_token_map or args.run_dir / "token_map.pt"
        if args.profile == "mechanism-smoke":
            token_map = torch.arange(recipe["text_config"]["vocab_size"], dtype=torch.long)
            temporary = token_path.with_name(f".{token_path.name}.{uuid.uuid4().hex}")
            torch.save(token_map, temporary)
            temporary.replace(token_path)
        else:
            token_map = torch.load(token_path, map_location="cpu", weights_only=True)
            text = recipe["text_config"]
            if (
                token_map.dtype != torch.int64
                or token_map.shape != (text["vocab_size"],)
                or token_map.min() != 0
                or token_map.unique().numel() != text["engram_compressed_vocab_size"]
                or token_map.max() + 1 != text["engram_compressed_vocab_size"]
            ):
                raise ValueError("Token map does not match released tokenizer compression")
        os.chdir(Path(__file__).resolve().parents[2])
        environment = dict(os.environ)
        if args.cp > 1 and torch.cuda.get_device_capability()[0] < 10:
            environment.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
