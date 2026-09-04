# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Torch-free helpers for persisting full-iteration CUDA graphs (Foundry-style SAVE/LOAD).

Everything in this module is pure Python so it can be unit-tested without a GPU or torch.
The torch-dependent orchestration lives in ``megatron.core.full_cuda_graph_persist``.

Terminology (mirrors the Foundry paper, arXiv:2604.06664):

* SAVE: a normal run (warmup + capture) that additionally serializes the captured graph, the
  kernel binaries and the execution context needed to rebuild it in a fresh process.
* LOAD: a fresh process that skips warmup + capture and rebuilds the executable graph from the
  archive. Every device address embedded in the graph must be valid again, which is why the
  allocation region base/size and the phase cursor offsets are part of the manifest.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
from typing import Any, Callable, Iterable, Optional

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "fullcg_manifest.json"
ALLOC_ALIGNMENT = 2 * 1024 * 1024  # Foundry's kAllocAlignment (2 MiB)


# --------------------------------------------------------------------------------------
# Nested structure <-> flat tensor list
# --------------------------------------------------------------------------------------

_SCALAR_TYPES = (bool, int, float, str)


def flatten_struct(obj: Any, is_tensor: Callable[[Any], bool], tensors: Optional[list] = None):
    """Flatten a nested (list/tuple/dict) structure into a JSON schema plus a flat tensor list.

    The forward_backward_func result and StaticBufferLoader buffers are nested containers of
    CUDA tensors and plain Python scalars. Foundry only understands a flat list of tensors, so we
    keep the container layout in ``schema`` and hand Foundry the flat list.

    Returns:
        (schema, tensors) where ``schema`` is JSON-serializable and ``tensors`` is the flat list
        (appended to if provided).
    """
    if tensors is None:
        tensors = []
    if is_tensor(obj):
        idx = len(tensors)
        tensors.append(obj)
        return {"t": "tensor", "i": idx}, tensors
    if obj is None:
        return {"t": "none"}, tensors
    if isinstance(obj, _SCALAR_TYPES):
        return {"t": "scalar", "v": obj, "py": type(obj).__name__}, tensors
    if isinstance(obj, (list, tuple)):
        items = []
        for item in obj:
            sch, tensors = flatten_struct(item, is_tensor, tensors)
            items.append(sch)
        return {"t": "list" if isinstance(obj, list) else "tuple", "items": items}, tensors
    if isinstance(obj, dict):
        keys, items = [], []
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"Only str dict keys are supported in graph results, got {type(k)}")
            sch, tensors = flatten_struct(v, is_tensor, tensors)
            keys.append(k)
            items.append(sch)
        return {"t": "dict", "keys": keys, "items": items}, tensors
    raise TypeError(
        f"Unsupported object of type {type(obj)} in CUDA graph result/static buffers. "
        "Only tensors, None, bool/int/float/str and list/tuple/dict containers can be persisted."
    )


def unflatten_struct(schema: dict, tensors: list) -> Any:
    """Inverse of :func:`flatten_struct`."""
    kind = schema["t"]
    if kind == "tensor":
        return tensors[schema["i"]]
    if kind == "none":
        return None
    if kind == "scalar":
        v = schema["v"]
        py = schema.get("py")
        if py == "bool":
            return bool(v)
        if py == "int":
            return int(v)
        if py == "float":
            return float(v)
        return v
    if kind in ("list", "tuple"):
        items = [unflatten_struct(s, tensors) for s in schema["items"]]
        return items if kind == "list" else tuple(items)
    if kind == "dict":
        return {k: unflatten_struct(s, tensors) for k, s in zip(schema["keys"], schema["items"])}
    raise ValueError(f"Unknown schema node type {kind!r}")


def count_tensors(schema: dict) -> int:
    """Number of tensor leaves referenced by a schema."""
    kind = schema["t"]
    if kind == "tensor":
        return 1
    if kind in ("list", "tuple", "dict"):
        return sum(count_tensors(s) for s in schema["items"])
    return 0


# --------------------------------------------------------------------------------------
# Fingerprint (archive invalidation)
# --------------------------------------------------------------------------------------

# Anything whose value legitimately differs between SAVE and LOAD runs of the *same* job config
# must be excluded: paths, ranks/addresses, timing, logging, checkpoint bookkeeping, seeds
# (seeds affect RNG contents, not graph structure; RNG is rebound at LOAD), and optimizer
# hyper-parameters / schedules (applied outside the captured graph). Whole-key regexes rather than
# substrings so that e.g. ``moe_router_load_balancing_type`` (contains "load") is kept.
FINGERPRINT_DENY_REGEXES = (
    r"^(load|save|pretrained_checkpoint|finetune|no_load_.*|no_save_.*|ckpt_.*|.*_ckpt.*|.*checkpoint.*)$",
    r"^(.*_dir|.*_path|.*_paths|.*_file|split|data_.*|.*_data_.*|mock_data|tokenizer_model|.*_tokenizer)$",
    r"^(tensorboard.*|wandb.*|log_.*|.*_log.*|one_logger.*|timing_.*|.*_timer.*|profile.*|nsys.*|use_pytorch_profiler|record_memory_history|memory_snapshot_path)$",
    r"^(rank|local_rank|world_size|node_rank|master_.*|.*_port|.*_addr|.*_backend|.*_timeout.*|.*_timeout)$",
    r"^(.*_interval|exit_.*|iteration|curr_iteration|consumed_.*|skipped_.*|train_iters|train_samples|eval_iters|eval_interval|.*_warmup_samples|.*_warmup_iters|.*_warmup_fraction|.*_warmup_init)$",
    r"^(seed|.*_seed|data_parallel_random_init)$",
    r"^(lr|min_lr|lr_.*|weight_decay.*|start_weight_decay|end_weight_decay|clip_grad|adam_.*|sgd_momentum|.*_lr_mult|loss_scale.*|initial_loss_scale|min_loss_scale|hysteresis)$",
    r"^(num_workers|straggler.*|.*_signal.*|async_.*|yaml_cfg|app_tag_run_.*|job_name|rerun_.*|error_injection.*|kill_switch_file)$",
    r"^(cuda_graph_persist_.*|cuda_graph_archive_dir|inprocess_.*|enable_.*_restart)$",
)
_FINGERPRINT_DENY = tuple(re.compile(r) for r in FINGERPRINT_DENY_REGEXES)

# Kept even if a deny regex matches (they change graph structure / allocation sizes).
FINGERPRINT_FORCE_INCLUDE = (
    "seq_length",
    "micro_batch_size",
    "global_batch_size",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "data_parallel_size",
    "num_layers",
    "hidden_size",
    "num_attention_heads",
    "num_experts",
    "moe_router_topk",
    "moe_router_load_balancing_type",
    "moe_token_dispatcher_type",
    "moe_flex_dispatcher_backend",
    "attention_backend",
    "fp8",
    "fp8_recipe",
    "fp8_format",
    "fp4",
    "bf16",
    "fp16",
    "optimizer",
    "use_distributed_optimizer",
    "cuda_graph_impl",
    "cuda_graph_warmup_steps",
    "deterministic_mode",
    "attention_dropout",
    "hidden_dropout",
)


def _is_json_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def is_fingerprint_denied(key: str) -> bool:
    return any(r.match(key) for r in _FINGERPRINT_DENY)


def select_fingerprint_fields(
    args_dict: dict,
    force_include: Iterable[str] = FINGERPRINT_FORCE_INCLUDE,
) -> dict:
    """Pick the subset of ``vars(args)`` that determines graph structure/addresses.

    Values are coerced to JSON-friendly representations (``repr`` for anything exotic).
    """
    force = set(force_include)
    out = {}
    for k in sorted(args_dict):
        if k.startswith("_"):
            continue
        if k not in force and is_fingerprint_denied(k):
            continue
        v = args_dict[k]
        if _is_json_scalar(v):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x if _is_json_scalar(x) else repr(x) for x in v]
        elif isinstance(v, dict):
            out[k] = {str(kk): (vv if _is_json_scalar(vv) else repr(vv)) for kk, vv in v.items()}
        else:
            out[k] = repr(v)
    return out


def compute_fingerprint(fields: dict) -> str:
    """Stable sha256 over a JSON-canonicalized dict."""
    blob = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def diff_fingerprint_fields(saved: dict, current: dict, max_items: int = 20) -> list[str]:
    """Human-readable list of differing keys (for fail-open diagnostics)."""
    diffs = []
    for k in sorted(set(saved) | set(current)):
        if saved.get(k, "<missing>") != current.get(k, "<missing>"):
            diffs.append(f"{k}: saved={saved.get(k, '<missing>')!r} current={current.get(k, '<missing>')!r}")
            if len(diffs) >= max_items:
                diffs.append("...")
                break
    return diffs


# --------------------------------------------------------------------------------------
# Pointer extraction from a Foundry graph JSON
# --------------------------------------------------------------------------------------


def iter_words(hex_str: str, word_bytes: int = 8, step: int = 4):
    """Yield (offset, little-endian unsigned int) for every ``step``-aligned window.

    Kernel argument buffers are opaque (cuBLAS/cuDNN pack pointers into structs), so we look at
    every 4-byte aligned 8-byte window rather than trusting declared parameter boundaries.
    """
    if not hex_str:
        return
    raw = bytes.fromhex(hex_str)
    n = len(raw)
    for off in range(0, n - word_bytes + 1, step):
        yield off, int.from_bytes(raw[off : off + word_bytes], "little", signed=False)


def extract_candidate_pointers(graph_json: dict, lo: int, hi: int) -> dict[int, list[str]]:
    """Scan every node's argument bytes for values that fall inside ``[lo, hi)``.

    Returns ``{pointer_value: [where, ...]}`` where ``where`` is a short provenance string
    (``node<id>:param<i>@<off>``). Only values inside the allocation region are considered
    pointers; a region base like 0x5000_0000_0000 makes accidental collisions negligible.
    """
    found: dict[int, list[str]] = {}

    def _add(val: int, where: str):
        if lo <= val < hi:
            found.setdefault(val, []).append(where)

    for node in graph_json.get("nodes", []):
        ntype = node.get("type")
        nid = node.get("id")
        params = node.get("params", {}) or {}
        if ntype == "KernelNode":
            for p in params.get("kernelParams", []) or []:
                for off, val in iter_words(p.get("value_hex", "")):
                    _add(val, f"node{nid}:param{p.get('index')}@{off}")
            for off, val in iter_words(params.get("extra_argBuffer_hex", "")):
                _add(val, f"node{nid}:argbuf@{off}")
        elif ntype == "MemcpyNode":
            for key in ("srcDevice", "dstDevice"):
                if key in params:
                    _add(int(params[key]), f"node{nid}:{key}")
        elif ntype == "MemsetNode":
            if "dst" in params:
                _add(int(params["dst"]), f"node{nid}:dst")
    return found


# --------------------------------------------------------------------------------------
# Segments / intervals
# --------------------------------------------------------------------------------------


def segments_from_snapshot(snapshot: list[dict], device: Optional[int] = None) -> list[tuple[int, int, str]]:
    """Normalize ``torch.cuda.memory_snapshot()`` into ``[(address, size, pool_tag)]``.

    ``pool_tag`` is "private" for graph private pools (segment_pool_id != (0, 0)) else "default".
    ``device`` filters to one CUDA device (the snapshot covers all devices of the process).
    Torch-free: takes the already-materialized list of dicts.
    """
    segs = []
    for seg in snapshot:
        if device is not None and "device" in seg and int(seg["device"]) != int(device):
            continue
        addr = int(seg["address"])
        size = int(seg.get("total_size", seg.get("size", 0)))
        pool_id = seg.get("segment_pool_id", (0, 0))
        if isinstance(pool_id, (list, tuple)):
            private = tuple(pool_id) != (0, 0)
        else:
            private = bool(pool_id)
        segs.append((addr, size, "private" if private else "default"))
    segs.sort()
    return segs


def find_segment(segments: list[tuple[int, int, str]], ptr: int):
    """Return the segment tuple containing ``ptr`` or None. ``segments`` must be sorted."""
    if not segments:
        return None
    starts = [s[0] for s in segments]
    i = bisect.bisect_right(starts, ptr) - 1
    if i < 0:
        return None
    addr, size, _ = segments[i]
    return segments[i] if addr <= ptr < addr + size else None


def classify_pointers(
    pointers: Iterable[int],
    segments: list[tuple[int, int, str]],
    init_end_addr: int,
    capture_start_addr: int,
    scratch_end_addr: int = 0,
) -> dict[str, list]:
    """Bucket graph-referenced pointers by which allocation phase created their segment.

    * ``scratch``: segment allocated in the comm-init scratch prefix (below ``scratch_end_addr``).
      Addresses there are NOT reproducible between SAVE and LOAD (the prefix exists precisely to
      absorb non-deterministic NCCL init); a graph pointer into it is a bug (a caching-allocator
      block created before the scratch jump and reused later). ``after_comm_init`` empties the
      cache before the jump to prevent this.
    * ``init``: segment allocated before the first training iteration -> LOAD recreates it by
      running the same initialization sequence.
    * ``warmup``: segment allocated during warmup iterations (after init, before capture) ->
      LOAD does *not* run warmup, so these must be physically mapped from the manifest.
      StaticBufferLoader buffers land here by construction; anything else is a lazily created
      persistent object that should ideally move to init (see warnings).
    * ``capture``: segment allocated inside the capture window -> Foundry's allocator-event
      replay recreates it.
    * ``unmanaged``: pointer not inside any caching-allocator segment (NVSHMEM heap, NCCL
      buffers, foreign allocations). Must be deterministic by other means.
    """
    segments = sorted(segments)
    out: dict[str, list] = {"scratch": [], "init": [], "warmup": [], "capture": [], "unmanaged": []}
    for ptr in sorted(set(pointers)):
        seg = find_segment(segments, ptr)
        if seg is None:
            out["unmanaged"].append((ptr, None))
            continue
        addr = seg[0]
        if scratch_end_addr and addr < scratch_end_addr:
            out["scratch"].append((ptr, seg))
        elif addr >= capture_start_addr or seg[2] == "private":
            out["capture"].append((ptr, seg))
        elif addr >= init_end_addr:
            out["warmup"].append((ptr, seg))
        else:
            out["init"].append((ptr, seg))
    return out


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent ``(addr, size)`` intervals."""
    ivs = sorted((int(a), int(s)) for a, s in intervals if s > 0)
    merged: list[list[int]] = []
    for a, s in ivs:
        if merged and a <= merged[-1][0] + merged[-1][1]:
            end = max(merged[-1][0] + merged[-1][1], a + s)
            merged[-1][1] = end - merged[-1][0]
        else:
            merged.append([a, s])
    return [(a, s) for a, s in merged]


def required_intervals(classified: dict[str, list]) -> list[tuple[int, int]]:
    """Segments LOAD must map explicitly: everything referenced from the warmup phase."""
    return merge_intervals((seg[0], seg[1]) for _, seg in classified["warmup"])


def align_up(x: int, alignment: int = ALLOC_ALIGNMENT) -> int:
    return (x + alignment - 1) & ~(alignment - 1)


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------


def stage_archive_names(stage: str) -> dict[str, str]:
    """File names used inside an archive directory for one stage (training/validation)."""
    # Not ``graph_<stage>.json``: Foundry's save_graph_manifest() globs ``graph_*.json`` and
    # expects an integer index in the file name.
    return {
        "graph_json": f"fullcg_{stage}.json",
        "graph_bin": f"fullcg_{stage}.cugraph",
    }


def build_manifest(
    fingerprint: str,
    fingerprint_fields: dict,
    region_base: int,
    region_size: int,
    rank_key: str,
    env: dict,
) -> dict:
    return {
        "version": MANIFEST_VERSION,
        "fingerprint": fingerprint,
        "fingerprint_fields": fingerprint_fields,
        "region": {"base": region_base, "size": region_size},
        "rank_key": rank_key,
        "env": env,
        "stages": {},
    }


def add_stage_to_manifest(
    manifest: dict,
    stage: str,
    result_schema: dict,
    static_buffers_schema: dict,
    num_tensors: int,
    generator_names: list[str],
    phase_offsets: dict,
    required: list[tuple[int, int]],
    pointer_report: dict,
    warnings: list[str],
) -> None:
    names = stage_archive_names(stage)
    manifest["stages"][stage] = {
        "graph_json": names["graph_json"],
        "graph_bin": names["graph_bin"],
        "result_schema": result_schema,
        "static_buffers_schema": static_buffers_schema,
        "num_tensors": num_tensors,
        "generator_names": generator_names,
        "phase_offsets": phase_offsets,
        "required_intervals": [[a, s] for a, s in required],
        "pointer_report": pointer_report,
        "warnings": warnings,
    }


def write_manifest(archive_dir: str, manifest: dict) -> str:
    os.makedirs(archive_dir, exist_ok=True)
    path = os.path.join(archive_dir, MANIFEST_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=repr)
    os.replace(tmp, path)
    return path


def read_manifest(archive_dir: str) -> Optional[dict]:
    path = os.path.join(archive_dir, MANIFEST_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def summarize_pointer_report(classified: dict[str, list]) -> dict:
    """Compact, JSON-friendly summary for the manifest / logs."""

    def _segs(items):
        segs = sorted({(seg[0], seg[1]) for _, seg in items if seg is not None})
        return [[a, s] for a, s in segs]

    return {
        "num_pointers": {k: len(v) for k, v in classified.items()},
        "scratch_segments": _segs(classified.get("scratch", [])),
        "init_segments": _segs(classified["init"]),
        "warmup_segments": _segs(classified["warmup"]),
        "capture_segments": _segs(classified["capture"]),
        "unmanaged_pointers": [hex(p) for p, _ in classified["unmanaged"]][:64],
    }
