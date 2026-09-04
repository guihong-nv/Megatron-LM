# Full-iteration CUDA graph persistence (Foundry SAVE/LOAD) — real training test guide

Branch: `fullcg-persist-foundry` (based on Megatron-LM `main` @ `4ec08fe`, 2026-09-03).
Foundry patch: `tools/foundry_patches/foundry_training_support.patch` (against Foundry `main` @ `4df02a0`).

This guide is the step-by-step protocol for the first GPU contact of the patch set. It goes from
"is the hook even active" to "LOAD reproduces the native run bitwise", then to the two
configurations that decide the next phase (EP with NVSHMEM, PP>1 with NCCL inside the graph).
Every stage has an explicit pass criterion and the commands to diagnose a failure.

> **State of the code:** the Python side is unit-tested but has never run on a GPU; the Foundry C++
> patch was written against the sources but not compiled. Expect small compile fixes in step 1.3
> and expect stage A/B to surface allocation-trajectory issues that no amount of desk review can
> find. That is what these stages are for.

---

## 0. What you are testing

| Term | Meaning |
|---|---|
| **native** | the regular `--cuda-graph-impl full_iteration` path: `cuda_graph_warmup_steps` eager iterations, then one capture, then replay. |
| **SAVE** | native + serialization: graph JSON/binary, kernel binaries, allocator events, static input buffers, result structure, RNG names, pointer provenance report → `<archive>/<rank_key>/`. |
| **LOAD** | fresh process: no warmup, no capture; rebuilds the executable graph from the archive and replays from its first iteration. Manifest / fingerprint / trajectory mismatches fall back to native (unless `--cuda-graph-persist-strict`); failures after memory has been mapped are fatal by design. |
| **archive** | `--cuda-graph-archive-dir`. One sub-dir per rank (`rank0/` …) by default. |

Phase 1 scope (this branch): graphs **without NCCL kernels inside** — TP=1, PP=1, DP=1; EP over
NVSHMEM (hybridep/DeepEP) is allowed. PP>1 or DP>1 puts NCCL send/recv / grad reduce-scatter inside
the captured region; the SAVE run then only produces a diagnostic archive (stage D).

---

## 1. Environment

### 1.1 Constraints

* **PyTorch 2.9 – 2.11.** Foundry re-implements `at::CUDAGeneratorState` methods with hidden
  visibility; PyTorch `main` changed the generator API (`CUDAGeneratorCaptureState`) and Foundry
  will not build against it. Check `python -c "import torch; print(torch.__version__)"` first.
* **CUDA driver ≥ 12.3** (graph edge data / `cuGraphAddDependencies_v2`), **≥ 12.4** recommended
  (`cuFuncGetParamInfo` used by Foundry's serializer).
* **`PYTORCH_CUDA_ALLOC_CONF` must not enable `expandable_segments`.** That allocator mode issues its
  own VMM calls and conflicts with the hook.
* The standard full-iteration requirements still apply: `--no-check-for-nan-in-loss-and-grad`,
  TE RNG tracker (auto-enabled), `TORCH_NCCL_AVOID_RECORD_STREAMS=1`, `NCCL_GRAPH_REGISTER=0`.
* Same container image for SAVE and LOAD. Any change of Megatron commit / PyTorch / TE / CUDA /
  NCCL / NVSHMEM / GPU SKU / model or parallel config invalidates the archive (fingerprint check,
  the LOAD log prints the differing keys).

### 1.2 Get the code

```bash
git clone https://github.com/guihong-nv/Megatron-LM.git -b fullcg-persist-foundry
cd Megatron-LM && export MLM=$PWD
```

### 1.3 Build Foundry with the training patch

```bash
git clone https://github.com/foundry-org/foundry.git && cd foundry
git checkout 4df02a025d08dd862a09c56b71716e8dbdc7f4d2
git apply $MLM/tools/foundry_patches/foundry_training_support.patch
# Foundry's pyproject pins torch==2.9.0; if your container has 2.10/2.11, relax the pin:
sed -i 's/"torch==2.9.0"/"torch>=2.9,<2.12"/' pyproject.toml
pip install --no-build-isolation -e .          # builds foundry.ops (torch extension) + libcuda_hook.so
python - <<'EOF'
import foundry, foundry.ops as ops, importlib.util, pathlib
so = pathlib.Path(importlib.util.find_spec("foundry.ops").origin).parent / "libcuda_hook.so"
print("hook:", so, so.exists())
print("A3 preallocate_intervals:", hasattr(ops, "preallocate_intervals"))
print("B1 bind_generator_state:", hasattr(ops.CUDAGraph, "bind_generator_state"))
EOF
export FOUNDRY_HOOK=$(python -c "import importlib.util,pathlib;print(pathlib.Path(importlib.util.find_spec('foundry.ops').origin).parent/'libcuda_hook.so')")
```

If the build fails, the likely spots are (in this order): `csrc/hook.cpp` `preallocate_intervals`
(driver typedef names must match the entry-table macros used elsewhere in the file),
`csrc/CUDAGraph.cpp` `bind_generator_state` (`ska::flat_hash_map::find/erase` on the
`intrusive_ptr` key), `include/CUDAGraph.h` includes. Each is self-contained; fix and re-run
`pip install`.

### 1.4 Common environment for every run

```bash
export LD_PRELOAD=$FOUNDRY_HOOK                 # every rank; torchrun inherits it
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_GRAPH_REGISTER=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ARCHIVE=/nvme/fullcg_archive              # local NVMe, ~1-3 GB per rank
```

> `LD_PRELOAD` in the launcher environment is not enough by itself: the hook must also be told
> the region. The patch does that in `initialize.py` when `--cuda-graph-persist-mode != none`;
> a 1 MiB probe allocation verifies the hook is intercepting and aborts (strict) or falls back
> (non-strict) otherwise — look for `probe allocation ... outside the allocation region`.

### 1.5 Baseline model config (single GPU, ~1 minute per run)

```bash
COMMON=(
  --num-layers 8 --hidden-size 1024 --num-attention-heads 16
  --seq-length 2048 --max-position-embeddings 2048
  --micro-batch-size 2 --global-batch-size 8          # 4 microbatches -> exercises the schedule
  --train-iters 40 --lr 1e-4 --lr-decay-style constant --min-lr 1e-4
  --bf16 --use-distributed-optimizer
  --tokenizer-type NullTokenizer --vocab-size 32000 --mock-data --num-workers 0
  --eval-iters 0 --eval-interval 100000               # no validation graph in stage A/B
  --log-interval 1 --seed 1234
  --deterministic-mode                                # bitwise-reproducible kernels (no atomics)
  --transformer-impl transformer_engine
  --cuda-graph-impl full_iteration --cuda-graph-warmup-steps 3
  --no-check-for-nan-in-loss-and-grad
  --attention-dropout 0.0 --hidden-dropout 0.0        # stage A/B: no RNG in the graph
)
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
run() { torchrun --nproc_per_node ${NPROC:-1} pretrain_gpt.py "${COMMON[@]}" "$@"; }
```

Keep this config identical across all runs of a stage. `--seed`, paths, `--train-iters`, LR /
schedule settings and checkpoint options do not affect the fingerprint; shapes, parallel sizes,
precision, optimizer type and the persistence region do.

**Every run in stages B–F must load a checkpoint** (the SAVE run too): checkpoint loading allocates
device temporaries, so a SAVE run that started from scratch and a LOAD run that resumed from a
checkpoint would have different allocation trajectories and fail the cursor check. Produce a seed
checkpoint once:

```bash
export CKPT=/nvme/fullcg_ckpt
NPROC=1 run --save $CKPT/seed --save-interval 1 --train-iters 1 2>&1 | tee seed.log   # native, 1 iteration -> ckpt@1
```

---

## 2. Stage A — determinism of the allocation layout (2 SAVE runs, no LOAD yet)

Goal: prove that two independent processes of the same config produce byte-identical graphs. If
this fails, LOAD cannot work and there is no point going further.

```bash
NPROC=1 run --load $CKPT/seed --cuda-graph-persist-mode save --cuda-graph-persist-strict \
  --cuda-graph-archive-dir $ARCHIVE/A1 2>&1 | tee A1.log
NPROC=1 run --load $CKPT/seed --cuda-graph-persist-mode save --cuda-graph-persist-strict \
  --cuda-graph-archive-dir $ARCHIVE/A2 2>&1 | tee A2.log

python tools/fullcg_graph_inspect.py inspect $ARCHIVE/A1/rank0/fullcg_training.json
python tools/fullcg_graph_inspect.py diff  $ARCHIVE/A1/rank0/fullcg_training.json $ARCHIVE/A2/rank0/fullcg_training.json
python tools/fullcg_graph_inspect.py manifest-diff $ARCHIVE/A1/rank0 $ARCHIVE/A2/rank0
```

What to look at in `A1.log`:

```
[fullcg-persist] allocation region base=0x500000000000 size=... mode=save
[fullcg-persist] eagerly initialized N process groups
[fullcg-persist] comm scratch used <bytes> bytes, cursor moved to 0x40000000
[fullcg-persist] training_first_call cursor offset 0x...
[fullcg-timing] training warmup_iter_0: 12.3s      # <- the breakdown you wanted
[fullcg-timing] training warmup_iter_1: 1.1s
[fullcg-timing] training warmup_iter_2: 1.0s
CUDA graph capture done for training!!! (capture+instantiate 2.4s, 3 warmup iterations before it)
[fullcg-timing] training capture_and_instantiate: 2.4s
[fullcg-persist] graph JSON/binary written in ...
[fullcg-persist] SAVE training: <n> tensors, <k> required warmup intervals, report={"init": .., "warmup": .., "capture": .., "unmanaged": ..}
[fullcg-timing] training save: ...
```

Pass criteria:

| check | expected | if not |
|---|---|---|
| `inspect` verdicts | `NCCL: none inside the graph`, no `host memcpy`, no unsupported node types | NCCL present with TP=PP=DP=1 means a collective is issued inside `forward_backward_func` on the default group (e.g. loss averaging) → find it; it is a Phase-2 item. Host memcpy → find the `.cpu()` / pinned copy inside the captured region |
| `diff` | `"deterministic": true` | `nodes_with_arg_diffs > 0`: kernels listed under `arg_diff_by_kernel` receive different addresses in the two runs → something allocates non-deterministically before them. Compare `manifest-diff` phase offsets: differing `training_first_call` means init is non-deterministic (lazy NCCL comm, workspace, data-dependent allocation); equal offsets but differing args means the difference is inside warmup/capture |
| `manifest-diff` | `phase_offsets_equal` and `required_intervals_equal` | as above |
| `pointer_report.num_pointers.unmanaged` | 0 | pointers outside every caching-allocator segment (NVSHMEM heap, NCCL buffers, cudaMalloc from a library). Print them from the manifest and identify the owner with `cuda-gdb`/`compute-sanitizer` or by dumping `torch.cuda.memory_snapshot()` at capture |
| `pointer_report.num_pointers.scratch` | 0 | a graph pointer refers to memory allocated before the comm-scratch jump (non-reproducible addresses). `after_comm_init` empties the cache before the jump precisely to prevent this; if it still happens some tensor allocated during `init_process_group`/`initialize_model_parallel` stayed alive and is referenced by the graph — find it via the manifest's `scratch_segments`. LOAD refuses such archives |
| `warnings` in manifest | empty or only the expected "warmup-phase allocations that are not static input buffers" | anything about RNG at this stage means dropout is on |

Common first-contact failures at this stage:

* `probe allocation ... outside the allocation region` — hook not preloaded in the rank process
  (check `cat /proc/<pid>/maps | grep libcuda_hook`), or `set_allocation_region` failed
  (`[HOOK] ERROR: cuMemAddressReserve failed` — pick a different `--cuda-graph-persist-base-addr`,
  e.g. `0x600000000000`).
* `comm init consumed X bytes ... scratch_size` — raise `--cuda-graph-persist-scratch-size 4GB`.
* `TORCH_CHECK ... Graph contains unsupported node type!` at capture end — Foundry's serializer met a
  host/child/mem-alloc node. With TP=PP=DP=1 this points at a host callback inside the captured
  region (e.g. NCCL on the default group, `cudaLaunchHostFunc`). Dump the graph with
  `CUDA_GRAPH_DEBUG_DOT=1`-style tooling or bisect by disabling features.

## 3. Stage B — LOAD reproduces the reference run bitwise

The comparison has to be **graph replay vs graph replay at the same parameter state**. A naive
"native from scratch vs LOAD from scratch" comparison is not bitwise even when everything is
correct, because the native run's first `cuda_graph_warmup_steps` iterations are *eager*
executions whose last-ulp differences from the captured kernels propagate into every later
parameter update. So:

```bash
# reference = the SAVE run itself, resumed from ckpt@1: iterations 2,3,4 eager (warmup),
# 5 captured+replayed, 6.. replayed. It also writes a checkpoint every iteration.
MCORE_FULLCG_LOSS_DUMP=$ARCHIVE/B_ref NPROC=1 run --load $CKPT/seed --save $CKPT/ref --save-interval 1 \
  --cuda-graph-persist-mode save --cuda-graph-persist-strict --cuda-graph-archive-dir $ARCHIVE/B 2>&1 | tee B_ref.log

# LOAD run resumes from the reference's ckpt@4 (parameters after the three eager warmup steps)
# and replays iterations 5.. from the archive. Same parameters, same data order, graph vs graph.
mkdir -p $CKPT/at4 && cp -r $CKPT/ref/iter_0000004 $CKPT/at4/ && echo 4 > $CKPT/at4/latest_checkpointed_iteration.txt
MCORE_FULLCG_LOSS_DUMP=$ARCHIVE/B_load NPROC=1 run --load $CKPT/at4 \
  --cuda-graph-persist-mode load --cuda-graph-persist-strict --cuda-graph-archive-dir $ARCHIVE/B 2>&1 | tee B_load.log

# dumps carry the global iteration number, so they line up automatically (iterations 5..40)
python tools/fullcg_compare_loss.py dump $ARCHIVE/B_ref.rank0.jsonl $ARCHIVE/B_load.rank0.jsonl
python tools/fullcg_compare_loss.py log  B_ref.log B_load.log
```

Why the archive from the SAVE run is valid for a LOAD that resumed from a *different* checkpoint:
both processes run the same initialization code with the same tensor shapes (the checkpoint
iteration number only changes dataloader state on the CPU), so the allocation trajectory up to
`training_first_call` is identical — which the LOAD checks verify.

What to look at in `B_load.log`:

```
[fullcg-persist] loaded kernel binaries from .../B/rank0 in 1.2s
[fullcg-persist] training_first_call cursor offset 0x...      # must equal the SAVE value
[fullcg-persist] LOAD training: graph rebuilt in 3.1s (<n> tensors)
Restored full-iteration CUDA graph for training from archive
```

Pass criteria:

| check | expected |
|---|---|
| `compare_loss dump` | `"strict_pass": true` and `bitwise_equal == iterations_compared` (all common iterations 5..N) |
| `compare_loss log` | `"pass": true` (6-decimal losses and 3-decimal grad norms equal) |
| wall clock | time from process start to the first replayed iteration ≪ reference (reference = 3 warmup iterations + capture; LOAD = kernel-binary load + graph rebuild). Record both — this is the number that justifies the project |
| peak memory | `torch.cuda.max_memory_reserved()` (or the `mem-usage` log fields) not above the reference. LOAD never allocates the warmup activations, so it is usually *lower* |
| second LOAD | run the LOAD twice; both must pass and both must be bitwise identical to each other |

Failure decoding:

* `cursor at training first call is 0x... but SAVE recorded 0x...` — initialization allocated a
  different amount than at SAVE. Diff the `[fullcg-persist]` cursor lines of the two logs. Usual
  causes: one run loaded a checkpoint and the other did not; a lazily created NCCL communicator (a
  group not covered by `warm_up_process_groups()`); a library workspace created on first use (add a
  warm-up call in `after_comm_init`); a dataloader allocating on the device at construction. This
  error is fail-open: without `--cuda-graph-persist-strict` the run silently continues natively.
* `LOAD of stage training failed after the process state was modified; cannot fall back` — the
  checks passed but interval mapping / allocator replay / graph rebuild failed. Look at the
  wrapped message and the `[HOOK]` / `[REPLAY]` / `[foundry LOAD ERROR]` stderr lines just above.
  Typical: `[REPLAY] FATAL: Allocation address mismatch` (something allocated inside the region
  between `try_load()` and the replay — the design allocates nothing there, so a library did),
  `cuMemMap failed` in `preallocate_intervals` (an interval overlaps an existing mapping; compare
  `required_intervals` with `pointer_report.warmup_segments`), `function handle not found`
  (a kernel binary is missing from the archive — check `fatbin_entrypoint_packed.txt`).
* `graph consumes RNG but this Foundry build has no bind_generator_state()` — B1 not compiled in.
* Losses differ from iteration 5 on but LOAD ran — most likely a graph-referenced tensor whose
  *contents* (not address) are produced during warmup and not regenerated at LOAD (RoPE tables,
  `cu_seqlens`, grouped-GEMM pointer arrays written by a host→device copy outside the graph). The
  manifest warning "warmup-phase allocations that are not static input buffers" lists candidates;
  move their initialization to model construction. Run `inspect` and look for `host memcpy`.
* `an illegal memory access` at the first replay — a kernel dereferenced memory that is mapped but
  holds garbage (device-side metadata written outside the graph, see above) or a pointer into an
  unmapped range (an `unmanaged` pointer that moved). `compute-sanitizer --tool memcheck` on the
  LOAD run names the kernel; `inspect --json` maps it to a node id.

## 4. Stage C — RNG inside the graph (dropout / stochastic rounding)

Re-run stages A and B with `--attention-dropout 0.1 --hidden-dropout 0.1` (or the FP4
stochastic-rounding recipe you use in production). Expect in the manifest:
`phase_offsets.rng_consumed: true`, `generator_names` containing `default` and the TE tracker
names, and in `load.log` `rebound N generator states`. The bitwise criterion is unchanged: the
LOAD run must reproduce the native run from iteration 3 on — this is exactly the check that B1
(binding the archived generator slots to the live, checkpoint-restored generators) is correct.

The stage-B protocol already is a resume test; with dropout on, a LOAD whose RNG rebinding is
wrong will diverge from the reference at iteration 5 while a correct one stays bitwise — there is
no separate check needed.

## 5. Stage D — EP with NVSHMEM (single node, 8 GPUs)

Goal: the graph contains hybridep/NVSHMEM kernels but no NCCL; SAVE on all 8 ranks, LOAD on all 8.

Config changes (DeepSeek-style routing, TP=PP=DP=1, EP=8):

```bash
MOE=(
  --num-experts 32 --moe-router-topk 4 --expert-model-parallel-size 8
  --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep
  --moe-expert-rank-capacity-factor 1.5        # static a2a shapes (paged stash), required by full_iteration MoE
  --moe-grouped-gemm
)
NPROC=8 run "${MOE[@]}" --save $CKPT/moe_seed --save-interval 1 --train-iters 1 2>&1 | tee D_seed.log          # ckpt@1
# pass 1: records the lazily created comm buffers (expect "RE-RUN SAVE" + eager_init_pending: true)
NPROC=8 run "${MOE[@]}" --load $CKPT/moe_seed --cuda-graph-persist-mode save --cuda-graph-archive-dir $ARCHIVE/D1 2>&1 | tee D_pass1.log
# pass 2: the reference run; recipe applied before the first call, archive valid
MCORE_FULLCG_LOSS_DUMP=$ARCHIVE/D_ref NPROC=8 run "${MOE[@]}" --load $CKPT/moe_seed --save $CKPT/moe_ref --save-interval 1 \
  --cuda-graph-persist-mode save --cuda-graph-persist-strict --cuda-graph-archive-dir $ARCHIVE/D1 2>&1 | tee D_ref.log
grep -c '"eager_init_pending": false' $ARCHIVE/D1/rank0/fullcg_manifest.json   # must be 1
for r in 0 1 7; do python tools/fullcg_graph_inspect.py inspect $ARCHIVE/D1/rank$r/fullcg_training.json | grep -A20 verdicts; done
mkdir -p $CKPT/moe_at4 && cp -r $CKPT/moe_ref/iter_0000004 $CKPT/moe_at4/ && echo 4 > $CKPT/moe_at4/latest_checkpointed_iteration.txt
MCORE_FULLCG_LOSS_DUMP=$ARCHIVE/D_load NPROC=8 run "${MOE[@]}" --load $CKPT/moe_at4 \
  --cuda-graph-persist-mode load --cuda-graph-persist-strict --cuda-graph-archive-dir $ARCHIVE/D1 2>&1 | tee D_load.log
for r in 0 1 7; do python tools/fullcg_compare_loss.py dump $ARCHIVE/D_ref.rank$r.jsonl $ARCHIVE/D_load.rank$r.jsonl; done
```

Things specific to this stage:

* **SAVE is a two-pass procedure for MoE.** hybridep/DeepEP create their NVSHMEM buffers inside
  the first dispatch (warmup iteration 0), i.e. at a cursor position LOAD can never reproduce. The
  first SAVE run records the constructor calls into `eager_init_recipe.json`, flags the manifest
  `eager_init_pending: true` and logs `RE-RUN SAVE`. Run the identical SAVE command again: the
  recipe is applied right before the first training call in every subsequent run (SAVE and LOAD
  alike), the flag is cleared, and the archive is usable. `LOAD` refuses a pending archive with an
  explicit message. If the second pass still reports pending, a backend allocates something we do
  not wrap — the recipe file lists what was caught; add the missing constructor to
  `EAGER_INIT_TARGETS` in `full_cuda_graph_persist.py`.
* `inspect` should show `kernel_libraries` with `deepep/hybridep` and/or `nvshmem` entries and
  `NCCL: none`. If NCCL appears, the a2a fell back to an NCCL dispatcher or the router aux-loss
  reduction crosses EP ranks; both are Phase-2 items.
* `pointer_report.num_pointers.unmanaged > 0` is expected here: the NVSHMEM symmetric heap is not a
  caching-allocator segment. It is fine **only if** the heap base is identical in every run —
  verify with stage-A style `diff` between two SAVE runs (`nodes_with_arg_diffs` must be 0). If the
  heap moves, set `NVSHMEM_SYMMETRIC_HEAP_BASE`-style pinning (see NVSHMEM docs for your version)
  or reserve it inside the hook region.
* `[HOOK] ERROR ... cuIpc*` in the log → hybridep's NVLink path uses legacy IPC on hooked memory;
  report the exact message (the hook has an IPC compatibility layer; this tells us whether it covers
  hybridep's allocation path).
* LOAD calls `init_nvshmem_for_loaded_modules()` after hybridep initializes; if the log shows the
  NVSHMEM kernels failing at first replay (`invalid device symbol` / hangs), that call site is the
  first suspect (`full_cuda_graph_persist.py::_load_impl`).
* Cross-rank sharing (`--cuda-graph-persist-share-across-ranks`) is **off** on purpose: hybridep
  kernels carry the PE id in their arguments, so archives are per rank.

## 6. Stage E — PP=2 / DP=2 go/no-go for Phase 2 (diagnostic only)

Goal: measure what NCCL leaves inside a full-iteration graph. **LOAD is not expected to work**;
the SAVE archive is a diagnostic artifact.

```bash
NPROC=2 run --pipeline-model-parallel-size 2 --cuda-graph-persist-mode save \
  --cuda-graph-archive-dir $ARCHIVE/E_pp2 2>&1 | tee E_pp2.log
NPROC=2 run --cuda-graph-persist-mode save --cuda-graph-archive-dir $ARCHIVE/E_dp2 2>&1 | tee E_dp2.log   # DP=2

for a in E_pp2 E_dp2; do
  python tools/fullcg_graph_inspect.py inspect $ARCHIVE/$a/rank0/fullcg_training.json | grep -E "types|libraries|verdicts" -A3
  python tools/fullcg_graph_inspect.py comm $ARCHIVE/$a/rank0 --stage training
done
```

Two possible outcomes, both informative:

1. **Capture fails** with `Graph contains unsupported node type!` → NCCL inserted host nodes (or
   user-object-bearing nodes) into the captured graph. Phase 2 must drop those nodes at SAVE and
   re-capture the collective at LOAD (the design in `docs/fullcg_persist/README.md`, "Phase 2").
   Record the exact node types by adding a `printf` of `nodeType` in
   `CUDAGraph::analyze_captured_graph` before the `TORCH_CHECK`.
2. **Capture succeeds** and `inspect` reports N `nccl` kernel nodes → run `comm`: the recorded
   collective launches (`comm_trace_training.json`) should equal N. If they do, the placeholder
   scheme is straightforward (1 recorded collective ↔ 1 kernel node). If N is larger, NCCL split a
   collective into several kernels (protocol/algorithm specific) and the placeholder must cover a
   node *group*; if smaller, some collectives were coalesced (`ncclGroupStart/End`) or issued via a
   path the recorder does not see (`ProcessGroup` methods, functional collectives) — the trace
   tells you which ops are missing.

Do **not** set `MCORE_FULLCG_ALLOW_NCCL_NODES=1` on a multi-rank job unless you want to watch it
hang: the restored NCCL kernels would reference another process's communicator state.

## 7. Stage F — the real target (V4-Flash-like MoE, EP32)

Only after A–D pass. Same protocol as stage D at scale, with two additions:

* Compare the `[fullcg-timing]` lines of the SAVE run with the 30-minute number you measured:
  `warmup_iter_0` (JIT + lazy init), `warmup_iter_1/2` (steady iterations), `capture_and_instantiate`.
  LOAD removes all of them; what remains is `loaded kernel binaries in ...s` +
  `graph rebuilt in ...s`. Report both.
* Validation graph: keep `--eval-interval` as in production. The validation graph is captured
  lazily at the first eval, so SAVE must run at least `cuda_graph_warmup_steps + 1` eval
  iterations, and LOAD's cursor at the first validation call must match `validation_first_call`
  in the manifest (steady-state allocations between training and eval must be deterministic). If
  only validation fails to LOAD, the training graph still restores and validation falls back to
  native capture — acceptable, note it.

---

## 8. Reporting a result

Please collect, per stage:

1. `A*.log`, `load.log` (grep `fullcg`), the `fullcg_manifest.json` files, and the
   `inspect`/`diff`/`comm` outputs (they are JSON with `--json`).
2. The `compare_loss` JSON.
3. For failures inside Foundry: the `[HOOK]` / `[REPLAY]` / `[foundry ...]` stderr lines.
4. `nvidia-smi --query-gpu=name,driver_version --format=csv`, `python -c "import torch;print(torch.__version__, torch.version.cuda)"`,
   `pip show transformer_engine nvidia-nccl-cu12 | grep -i version`.

Things in this branch that are deliberately conservative and can be relaxed once A–D pass:
`stop_region_after_graph=False` (region stays active for the whole process), per-rank archives,
fail-open default, `MCORE_FULLCG_COMM_TRACE=1` (adds a few Python calls per collective during
capture only).
