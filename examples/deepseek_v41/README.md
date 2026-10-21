# DeepSeek-V4.1-Flash text training

This branch combines [#7678](https://github.com/NVIDIA/Megatron-LM/pull/7678) and
[#8223–#8227](https://github.com/NVIDIA/Megatron-LM/pull/8227): CSA2, single-pass
mHC, Engram, HybridStack, distributed Adam, packed CP, typed PP, and checkpointing.
The implementation retains attribution to Hongxiao Bai's
[#7224](https://github.com/NVIDIA/Megatron-LM/pull/7224), Guihong Li's earlier main
stack, and Li Tao's shared EP embedding work.

These recipes train the **BF16 text backbone and Engram from scratch**.
They do not reproduce released-model convergence, vision, DSpark, FP8/FP4 QAT,
quantized HF weight import, or the released optimizer recipe. Adam is an explicit
baseline. TP is one; attention CP uses contiguous packed layouts.
Automatic CLI CUDA graph capture, variable-token MoE capture and Engram capture
remain unsupported. The model's explicit graph APIs are separate.

## Profiles

`config_flash.json` is copied from the
[released configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json),
also previously tracked in #7503. Reference vision/DSpark fields are retained for
provenance; the launcher selects the text backbone. Every run records the source
SHA-256, original block selection, every changed field, command and Engram config.

| Profile | Blocks | Geometry and purpose |
| --- | --- | --- |
| `full` | All 40 | Released text/Engram dimensions and sharing schedule. |
| `depth-proxy` (default) | Original `[0,1,2,3,14,20,24,25]` | Only depth and layer indices change; widths, heads, experts and table capacities stay released size. |
| `mechanism-smoke` | Same eight selected blocks | Explicit synthetic test: H256, Q-rank128, O-rank32, expert I256, eight routed experts, vocab256, reduced Engram table rows and candidate capacity. This is not the depth-only proxy. |

Both faithful profiles retain H5120; 64 attention heads of dimension512;
Q-rank1280; eight output groups of rank1024; indexer32x128/TopK512; 384 routed
experts of intermediate width2304, top6 routing and one shared expert;
sqrt-softplus routing with correction biases and scaling1.5; clamped SwiGLU10;
four mHC streams with 20 Sinkhorn iterations; and both full-size Engram tables.
All profiles retain the mechanisms, attention head dimensions, top6/shared
routing, Engram n-gram orders2..4 and eight hash heads of width256.

| Proxy block | Original block | Attention role | Ratio | Engram |
| --- | --- | --- | --- | --- |
| 0 | 0 | SWA | 0 | |
| 1 | 1 | SWA | 0 | yes |
| 2 | 2 | Full | 2 | |
| 3 | 3 | Reuse | 2 | |
| 4 | 14 | Full | 2 | yes |
| 5 | 20 | Full; candidate producer | 1 | |
| 6 | 24 | Reindex | 1 | |
| 7 | 25 | Reuse | 1 | |

Each logical block is a physical `VE` pair (attention then MoE). The launcher
doubles source IDs and inserts ratio zero at E positions. Full-model physical
Full IDs are `[4,16,28,40]`, index producers are
`[4,16,28,40,48,56,64,72]`, and candidate source is40. Engram IDs are logical.
Remapped proxy blocks receive new depth indices and hash RNG seeds; proxy
checkpoints are not released-model checkpoints.

## Capacity

Reducing depth leaves both Engram tables, together 196.61 billion parameters.
These estimates count only routed experts and Engram tables, excluding
dense/shared weights, activations, communication, replication and stage imbalance.

| Profile | Parameter lower bound | BF16 weights GiB | Adam state and weights GiB | Ideal lower-bound GPUs at 180GiB and 75% occupancy |
| --- | ---: | ---: | ---: | ---: |
| `full` | 740.20 billion | 1378.72 | 12408.50 | 92 |
| `depth-proxy` | 305.33 billion | 568.72 | 5118.50 | 38 |

Adam uses 18 bytes per parameter in this estimate: BF16 weights, FP32 gradients,
master weights and two moments. These are capacity lower bounds, not verified
fits. The 128-GPU full and 64-GPU proxy plans below still require measurement.
A small allocation should use the separately labeled mechanism profile.
Nothing automatically reduces a faithful profile to fit.

## Exact token map

Use the Megatron **dev CI container**, with locked dependencies prepared once
before launch: FlashMLA, Transformer Engine, cuDNN Frontend and tokenizer/training
dependencies. The launcher uses `uv run --no-sync`, avoiding concurrent installs.

Build Engram identities from a local copy of the released tokenizer:

```bash
PYTHONPATH="$PWD" uv run --no-sync python examples/deepseek_v41/prepare_token_map.py \
  --tokenizer /shared/tokenizers/DeepSeek-V4.1-Flash \
  --output /shared/dsv41/token_map.pt
```

The helper requires 129280 raw and 99092 compressed identities, saves an int64
tensor, and records its hash/provenance. `--allow-download` explicitly permits a
named Hugging Face tokenizer download. Only `mechanism-smoke` generates an
identity map. Neither faithful profile can substitute that for the released map.

## Inspect and launch

Recipe generation needs only the Python standard library:

```bash
python3 examples/deepseek_v41/recipe.py --profile depth-proxy --dry-run \
  --run-dir /shared/dsv41/depth-proxy --nnodes 16 --nproc-per-node 4 \
  --pp 2 --ep 32 --global-batch-size 64 \
  --engram-token-map /shared/dsv41/token_map.pt

python3 examples/deepseek_v41/recipe.py --profile full --dry-run \
  --run-dir /shared/dsv41/full --nnodes 32 --nproc-per-node 4 \
  --pp 4 --ep 32 --global-batch-size 128 \
  --engram-token-map /shared/dsv41/token_map.pt
```

Unknown architecture overrides are rejected. Omitted `--data-path` means
synthetic mock data and is recorded in the manifest. For indexed text training,
provide `--data-path /shared/data/text_document` and
`--tokenizer-model /shared/tokenizers/DeepSeek-V4.1-Flash` together. The dataset
must use that tokenizer.

An eight-GPU synthetic test includes DP2/EP2/PP2/CP2, all attention roles, both
Engram insertions, nonzero indexer loss and model/optimizer checkpointing:

```bash
bash examples/deepseek_v41/train.sh --profile mechanism-smoke \
  --run-dir /shared/dsv41/mechanism --nproc-per-node 8 --pp 2 --cp 2 --ep 2 \
  --global-batch-size 8 --seq-length 1024 --steps 8 --save-interval 4

# Resume the same checkpoint and optimizer through total iteration16.
bash examples/deepseek_v41/train.sh --profile mechanism-smoke \
  --run-dir /shared/dsv41/mechanism --nproc-per-node 8 --pp 2 --cp 2 --ep 2 \
  --global-batch-size 8 --seq-length 1024 --steps 16 --save-interval 4
```

`--steps` is the target total iteration. Reusing the run directory resumes;
use a new directory for a fresh run. The checkpoint scheduler is preserved
explicitly so extending the target does not fail on horizon mismatches.
The generic entrypoint also supports `--use-checkpoint-args`: V4.1 checkpoints
embed Engram configuration, and `--engram-token-map` can relocate the same map.

Add `--full-recompute` for one-physical-layer uniform recomputation.
CP enables inter-document masking and contiguous attention/MLP layouts.
`CUDA_DEVICE_MAX_CONNECTIONS=1` is set only for pre-Blackwell CP, preserving an
explicit user setting.

## OCI Slurm

Submit from a shared checkout with the actual account and partition.
`MEGATRON_ROOT`, `CONTAINER_IMAGE` and `RUN_ROOT` are required shared paths.
`EXTRA_CONTAINER_MOUNTS` supplies tokenizer/data paths outside those roots.

```bash
export MEGATRON_ROOT=/shared/checkouts/Megatron-LM
export CONTAINER_IMAGE=/shared/images/megatron-ci-dev.sqsh
export RUN_ROOT=/shared/dsv41/mechanism
sbatch --account=<account> --partition=<partition> --nodes=2 --gpus-per-node=4 \
  examples/deepseek_v41/run_oci.slurm --profile mechanism-smoke \
  --pp 2 --cp 2 --ep 2 --global-batch-size 8 --steps 8
```

The wrapper starts one launcher per node and forwards rendezvous/rank/path
arguments. The OCI examples assume four GPUs per node. Faithful runs use the
dry-run plans above with matching node counts
and the exact token map.

## Checks and evidence

```bash
# No PyTorch or GPU needed.
python3 -m unittest discover -s examples/deepseek_v41 -p test_recipe.py -v

# Real Megatron parser and scheduler resume, in the dev environment.
PYTHONPATH="$PWD" uv run --no-sync python -m unittest discover \
  -s examples/deepseek_v41 -p test_megatron_parser.py -v
```

Inherited #8227 validation reports H100 forward/backward, checkpointing, PP/VPP,
packed CP and recomputation on small models. It does not qualify this branch's
new profiles or full-scale training. Current-run GPU evidence must be reported
separately with the tested commit and profile.
