# DeepSeek-V4.1-Flash text training

This branch combines [#7678](https://github.com/NVIDIA/Megatron-LM/pull/7678) and
[#8223–#8227](https://github.com/NVIDIA/Megatron-LM/pull/8227): CSA2, single-pass
mHC, Engram, HybridStack, distributed Adam, packed CP, typed PP, and checkpointing.
The implementation retains attribution to Hongxiao Bai's
[#7224](https://github.com/NVIDIA/Megatron-LM/pull/7224), Guihong Li's earlier main
stack, and Li Tao's shared EP embedding work.

These recipes train the **BF16 text backbone and Engram from scratch**.
They do not reproduce released-model convergence, vision, auxiliary MTP/DSpark
draft layers and heads, FP8/FP4 QAT, quantized HF weight import, or the released
optimizer recipe. Adam is an explicit baseline. TP is one; attention CP uses
contiguous packed layouts.
Automatic CLI CUDA graph capture, variable-token MoE capture and Engram capture
remain unsupported. The model's explicit graph APIs are separate.

## Numerical policy

The recipes enable `--moe-combine-in-fp32`. Routed expert outputs are gathered
in ascending expert order and accumulated in FP32. The shared expert is added
before one final cast to the layer input dtype. This requires unfused, dropless
AllToAll dispatch and avoids the previous BF16 atomic accumulation. Direct CLI
launches should pass this flag and `--moe-token-dispatcher-type alltoall`
explicitly; `DeepSeekV41Config` enables both by default for Python callers.
Other model configurations retain their existing default.

Checkpoint argument restoration preserves this saved numerical policy and its
required dispatcher. Legacy checkpoints without the field retain the caller's
requested policy. V4.1 also retains FP32 sine/cosine tables for both Q/K rotation
and inverse output rotation, while activations remain in the model dtype.

These changes align individual operations with the
[pinned reference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py).
The reference additionally applies FP8/FP4 activation quantization and uses
different compressor/indexer arithmetic. Full-model HF parity and whole-model
deterministic training require separate qualification; the existing CSA2
deterministic-mode restriction remains in place.

## Profiles

`config_flash.json` is copied from the
[released configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json),
also previously tracked in #7503. Reference vision and MTP/DSpark fields are
retained for provenance; the launcher selects the text backbone. The full profile
uses all 40 backbone compression ratios and omits the three trailing draft
entries. `num_nextn_predict_layers=3` is reference metadata, not instantiated
runtime layers. Every run records the source SHA-256, original block selection,
every changed field, command and Engram config.

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
master weights and two moments. These are capacity lower bounds. The eight-block
proxy completed two optimizer steps on 64 GB200 GPUs, and the full 40-block text
backbone plus Engram completed three optimizer steps and synthetic validation/test
on 128 GB300 GPUs, both at sequence length 1024. Longer-context capacity remains
unvalidated. Measurements are reported below with their tested source commits;
the large-capacity runs predate the numerical-policy changes above.
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

### Branch validation

Code commit `9e063d60e39ab85e326d2793083366637ed6dd50` was checked with:

- 22 CPU tests for checkpoint reconstruction, packed state, shared gradients,
  Engram communication, and compiler specialization; 16 standalone recipe tests;
  and three real-parser/configuration/scheduler tests.
- 15 focused CUDA tests on GB200, covering packed metadata, native/compiled
  parity, input gradients, train/eval transitions, and replay.
- Eight GB200 GPUs with TP1/PP2/CP2/EP2: four training iterations, evaluation,
  checkpoint save, then a fresh process loading checkpoint 4 and continuing
  through iteration 8. Model, optimizer and RNG state were restored; checkpoints
  6 and 8 and final validation/test completed. No iterations were skipped or NaN.
- Eight GB200 GPUs with TP1/PP4/CP1/EP2: two training iterations, checkpoint save,
  and validation/test. This split transports shared CSA2 KV/indexer/candidate
  state across a pipeline boundary.

The eight-GPU training checks used `mechanism-smoke`, sequence length 1024,
microbatch 1, global batch 8, BF16 and synthetic data. They qualify the execution
mechanisms and checkpoint lifecycle, not released-model convergence.
CPU numerical tests use native operations and CPU projection substitutes; they
do not stand in for CUDA kernel validation.

The same code commit also completed a `depth-proxy` run on 64 GB200 GPUs
(16 nodes, TP1/PP2/CP1/EP32), retaining the released widths, 384 routed experts,
both full-size Engram tables and the exact tokenizer map. With sequence length
1024, microbatch 1, global batch 64, BF16, Adam and synthetic data, both optimizer
steps completed and the job exited successfully. LM loss was 12.74057 then
6.677676, gradient norms were 13.595 and 15.954, and indexer auxiliary loss was
nonzero. No iterations were skipped or NaN.

The maximum across the two reported pipeline representatives was 98.9 GiB
allocated and 102.3 GiB reserved in the PyTorch allocator. These reports do not
measure every rank or include all device allocations. Device samples collected
about every two seconds across all 64 GPUs had a maximum observed usage of
128.8 GiB; sampling can miss transient peaks. Checkpoint writes and evaluation
were disabled for this capacity probe. It verifies short-context execution and
optimizer updates for the eight-block full-geometry proxy.

Source commit `0bddb43dc34385b0287a152751a030f623967231`, whose executable files
match `9e063d60`, also completed a `full` run on 128 GB300 GPUs (32 nodes,
TP1/PP4/CP1/EP32). All 40 text-backbone blocks, 384 routed experts, both full-size
Engram tables and the exact tokenizer map were retained. With sequence length
1024, microbatch 1, global batch 128, BF16, Adam and synthetic data, three
optimizer steps and final validation/test completed; the job exited successfully.

| Iteration | LM loss | Gradient norm | Indexer loss |
| --- | ---: | ---: | ---: |
| 1 | 12.79168 | 23.981 | 0.001379190 |
| 2 | 9.86920 | 37.260 | 0.001003445 |
| 3 | 12.86791 | 48.387 | 0.000564542 |

All values were finite, and no iterations were skipped or reported NaN. Loss was
non-monotonic; this short test does not establish convergence. Final validation
and test losses were 10.47138 and 10.70927. The first iteration took about 21 minutes,
including cold compilation; total job time was 25 minutes 7 seconds.

Across 92,064 device samples covering all 128 GPUs, the maximum observed usage
was 212.45 GiB. Sampling was approximately every two seconds and can miss
transient peaks. The maximum PyTorch allocator readings from the four reported
pipeline representatives were 166.19 GiB allocated and 176.97 GiB reserved;
these exclude other ranks and non-allocator device memory.

This run used NGC PyTorch 26.09, Transformer Engine 2.19, the repository's pinned
FlashMLA/frontend/DSL dependencies and a cuBLAS 13.8.1.7 overlay. Its preflight
passed the 15 CUDA tests and a four-GPU mechanism training/evaluation/checkpoint
test. Full-size checkpoint writes were disabled for the 128-GPU run. Checkpoint
save/resume remains separately validated on the smaller mechanism profile.
Full-size checkpoint IO, long-context capacity, convergence and production
throughput remain unvalidated; auxiliary MTP/DSpark draft layers and heads are
outside the backbone integration.

The GPU runs exposed and fixed evaluation failures from compiler-cache
specialization of packed indices and compressor inputs. The index metadata path
normalizes grad mode; the differentiable compressor retains autograd and uses
dynamic document dimensions. The global compiler cache limit is unchanged.

### Numerical regression and HF comparison

Source `894b8981a5d579585071e50c19091a5f5d9a7ee0` includes the FP32 expert
accumulation and rotary-table changes described above. Its production files are
identical to `e51f40c5`; intervening commits repair test fixtures. On OCI JHB
GB300, 53 distinct regression cases passed on both test workers, including
expert-parallel dispatch, forward/backward replay, empty-token gradients,
checkpoint argument restoration, and V4/V4.1 rotary dispatch. Three real-parser
tests and 16 standalone recipe tests also passed. Passing cases are recorded
across jobs `1200640`, `1201221` and `1201438`; affected test cases were rerun
after fixture repairs.

Job `1201438` compared actual production forwards with the official HF-hosted
inference implementation at revision
`314113f8a7dadb388215396ba5fbee2c610a04d6`. It used identical realized random
weights and token IDs, an eight-block reduced-width proxy, sequence length 128,
batch size one, and one compute GPU. All attention roles, both Engram sites,
mHC, and routed/shared experts were exercised. This does not test the released
pretrained checkpoint or full-depth forward parity.

Three target forwards produced bit-identical logits and recorded intermediate
tensors. Strict HF parity still **fails** at unchanged `atol=0.005`, `rtol=0.02`:

| Reference activation policy | All-token logit relative L2 | Last-token logit relative L2 |
| --- | ---: | ---: |
| Official FP8/FP4 quantize/dequantize enabled | 12.1127% | 11.4825% |
| Activation quantize/dequantize disabled for BF16 comparison | 4.5610% | 0.9302% |

Both comparisons use the actual target output; no target operations or expert
outputs are substituted. Bit-exact replay in this experiment does not establish
whole-model deterministic training. The numerical fixes remove two identified
differences but do not establish equivalence to the released inference recipe.

At publication, the post-change eight-GB300 training/checkpoint regression
(`1201918`, TP1/PP2/CP2/EP2, sequence length 512, full recomputation) is queued.
It will run two steps, save checkpoint 2, then resume in a fresh process through
step 3. The resume command deliberately omits the new accumulation flag so that
checkpoint restoration must recover it. These pending results are not included
in the completed checks above. A separate same-input component comparison is
prepared to localize the remaining HF differences after this allocation ends.
