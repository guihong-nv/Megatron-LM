# Full-iteration CUDA graph persistence for Megatron-LM (Foundry-backed SAVE/LOAD)

Patch set produced against:

* **Megatron-LM** `main` @ `4ec08fec74f8a6abf8e455b7f77ce99f29768e29` (2026-09-03)
* **Foundry** `main` @ `4df02a025d08dd862a09c56b71716e8dbdc7f4d2` (2026-08-26)

Goal (Phase 1 of the plan): with `--cuda-graph-impl full_iteration`, run warmup + capture once
(**SAVE**), then start fresh processes that rebuild the executable graph from the archive and
skip warmup + capture entirely (**LOAD**). Target config for Phase 1: graphs with **no NCCL
kernels inside** (TP=1, PP=1, DP=1; EP over NVSHMEM/hybridep is fine). PP/DP>1 puts NCCL
send/recv and grad reduce-scatter inside the graph and is Phase 2 (hybrid re-capture).

## Files

```
(this branch)                        Megatron-LM changes are committed on branch fullcg-persist-foundry
tools/foundry_patches/foundry_training_support.patch   git diff vs Foundry main (apply with `git apply`)
full_cuda_graph_persist.py           new: megatron/core/full_cuda_graph_persist.py
full_cuda_graph_persist_utils.py     new: megatron/core/full_cuda_graph_persist_utils.py
full_cuda_graph_comm_trace.py        new: megatron/core/full_cuda_graph_comm_trace.py (Phase-2 SAVE half)
fullcg_graph_inspect.py              new: tools/fullcg_graph_inspect.py (Phase-0 diagnostics CLI)
test_full_cuda_graph_persist_utils.py new: tests/unit_tests/... (9 tests, pass without GPU/torch)
test_fullcg_graph_inspect.py         new: tests/unit_tests/... (6 tests, pass without GPU/torch)
```

### Megatron side

| file | change |
|---|---|
| `megatron/core/full_cuda_graph_persist_utils.py` (new) | torch-free helpers: nested-struct <-> flat tensor list + JSON schema; archive fingerprint (denylist over `vars(args)` + forced shape/parallelism keys + env versions); pointer scan of Foundry graph JSON (`value_hex`, `extra_argBuffer_hex`, memcpy/memset addresses, 4-byte stride 8-byte windows); segment classification into init / warmup / capture / unmanaged from `torch.cuda.memory_snapshot()`; `required_intervals`; manifest I/O |
| `megatron/core/full_cuda_graph_persist.py` (new) | `PersistConfig.from_args` (env overrides `MCORE_FULLCG_*`); `FullCudaGraphPersistence` singleton: `early_init` (region + hook probe), `after_comm_init` (eager PG init, LOAD kernel-binary load, jump to scratch boundary), `mark_stage_first_call`, `new_graph`/`capture_context`, `save` (Foundry `graph.save(output_tensors=result+static buffers)`, `pack_fatbins_to_folder`, pointer classification, manifest), `try_load` = pure checks (fingerprint, cursor trajectory, scratch boundary, files) → all-ranks agreement (fail-open unless strict) → apply (map warmup intervals, `fdry.CUDAGraph.load`, rebuild result + static buffers, RNG rebind; failures here are fatal because the process state is already modified); `warm_up_process_groups()`; `after_comm_init` empties the caching allocator before the scratch jump so no pre-jump (non-reproducible) block can be reused later |
| `megatron/core/full_cuda_graph.py` | `FullCudaGraphWrapper.__call__`: LOAD branch **before** `data_read` (installs static buffers at archived addresses); capture uses `persist.new_graph()/capture_context()`; SAVE after capture. Unchanged behaviour when persistence is off |
| `megatron/core/transformer/transformer_config.py` | 7 new fields -> auto CLI flags: `--cuda-graph-persist-mode {none,save,load}`, `--cuda-graph-archive-dir`, `--cuda-graph-persist-base-addr`, `--cuda-graph-persist-region-size`, `--cuda-graph-persist-scratch-size`, `--cuda-graph-persist-strict`, `--cuda-graph-persist-share-across-ranks` |
| `megatron/training/initialize.py` | after `torch.cuda.set_device`: `init_persistence(args).early_init()`; after `initialize_model_parallel`: `after_comm_init()` |
| `megatron/core/full_cuda_graph_comm_trace.py` (new) | `CommTraceRecorder`: during capture (SAVE) wraps `torch.distributed.{all_reduce, reduce_scatter_tensor, all_gather_into_tensor, all_to_all(_single), broadcast, send/recv/isend/irecv, batch_isend_irecv, ...}` and records op / group ranks / buffer ptr+shape+dtype / peers / async flag in issue order -> `comm_trace_<stage>.json`. This is the semantic call sequence Phase 2 needs to re-capture NCCL collectives at LOAD. `MCORE_FULLCG_COMM_TRACE=0` disables |
| `tools/fullcg_graph_inspect.py` (new) | `inspect` (node types, kernel library attribution incl. demangling, NCCL / host-memcpy / RNG / PDL flags, pointer report + verdicts), `diff` (two SAVE runs: structure + per-node kernel-argument diffs = determinism check), `manifest-diff` (phase offsets / required intervals), `comm` (recorded collectives vs NCCL kernel nodes) |
| `full_cuda_graph.py` / `full_cuda_graph_persist.py` timing | `[fullcg-timing]` log lines + `stages.<stage>.timing` in the manifest: each warmup iteration, `capture_and_instantiate`, `save` — the breakdown of the 30 minutes. LOAD refuses archives whose graph contains NCCL kernel nodes (`MCORE_FULLCG_ALLOW_NCCL_NODES=1` overrides for experiments) |

### Foundry side (C++; **not compiled here**, no CUDA toolchain in the authoring environment)

| id | change | why |
|---|---|---|
| A0 | `hook.cpp`: allocation-region state `thread_local` -> process-global + `std::recursive_mutex` (`FOUNDRY_LOCK_STORAGE()` in `cuMemAlloc_v2/AllocPitch_v2/Free_v2/AddressReserve` and all `foundry::` region APIs) | PyTorch runs backward on autograd worker threads; every allocation made while capturing the backward half of a training graph bypassed the region (real `cudaMalloc`, unrecorded). Fatal for full-iteration training; invisible in inference |
| A3 | `hook.cpp/.h`, `binding.cpp`, `ops.pyi`: `preallocate_intervals([(addr,size),...]) -> bool` | LOAD must physically map warmup-phase segments the graph references (StaticBufferLoader buffers, lazily created workspaces, RNG extragraph tensors). Skips already-mapped ranges; moves the cursor past the highest mapped range so pre-replay allocations cannot collide. Python has a ctypes fallback (drives the hooked `cuMemAlloc_v2` cursor-walk) when the patch is absent |
| B1 | `CUDAGraph.h/.cpp`, `binding.cpp`, `ops.pyi`: `set_generator_name(gen, name)` (SAVE), `bind_generator_state(name, gen)` (LOAD); `save()` records `name`, `seed_extragraph_ptr`, `offset_extragraph_ptr`; `load()` keeps `GeneratorRecord`s and, when the archive has extragraph addresses, aliases them with `from_blob` instead of allocating placeholder tensors (an `at::empty` there would move the region cursor before the capture-window replay); `register_generator_state(state)` no longer clobbers an existing `wholegraph_increment` | Foundry LOAD created fresh generator states from the archived seed with offset 0 and relied on re-allocating the extragraph tensors at the same trajectory point. In Megatron those tensors live in a warmup segment LOAD never allocates, and the states were never bound to the checkpoint-restored generators, so dropout / stochastic-rounding streams would restart from the SAVE seed on every restart. Now the live state's extragraph tensors alias the archived addresses, so `replay_prologue()` writes the live seed/offset there. Slots with `wholegraph_increment == 0` only update bookkeeping (their addresses may not be mapped) |

| A7 | `metadata.h`, `CUDAGraph.cpp` (analyze/save/load), `CUDAGraphParallel.cpp`: `GraphDependency` carries `CUgraphEdgeData` (`type`, `from_port`, `to_port`); saved as optional JSON fields on non-default edges; restored through `cuGraphAddDependencies(_v2)` with edge data | Programmatic-dependent-launch edges (DeepGEMM, TE, cuBLASLt on Hopper/Blackwell) were flattened to full dependencies: correct but loses the overlap PDL buys |

Not in this patch (documented gaps): host/child/mem-alloc nodes are still rejected by the
serializer (so a PP/DP>1 SAVE fails loudly at `analyze_captured_graph`, which is the intended
Phase-2 go/no-go test); the binary `.cugraph` format carries neither the new generator fields
nor edge data (the JSON path, which this integration uses, has both).

## Phase 2 (NCCL inside the graph) — what exists now and what is left

SAVE half (in this patch): `comm_trace_<stage>.json` records every collective issued during
capture; `inspect` attributes kernel nodes to libraries and `comm` compares recorded launches with
the NCCL kernel-node count. Run a PP=2 (or DP=2) SAVE and look at these before writing any
LOAD-side code — it tells you exactly which node types / how many NCCL launches must be handled.

LOAD half (not written): (1) SAVE replaces NCCL kernel nodes (identified by binary hash of
libnccl / `ncclDevKernel_*` names) with `EmptyNode` placeholders that keep the in/out edges and
carry the `comm_trace` sequence number; NCCL's host nodes / user objects are dropped from the
archive. (2) LOAD, after communicators are eagerly initialized, re-captures each recorded
collective on a side stream into a small `torch.cuda.CUDAGraph(keep_graph=True)` with buffers
rebuilt by `from_blob` at the recorded addresses, then `cuGraphAddChildGraphNode` clones it into
the placeholder's position (`cuGraphNodeGetDependencies` / `cuGraphNodeGetDependentNodes` of the
placeholder, then `cuGraphDestroyNode`) and instantiates the parent once at the end. Needs a
Foundry `load()` variant that stops before instantiate plus a `splice_child_graph(node_id,
child_cuda_graph_ptr)` method. Open question to validate on GPU first: whether
`cuGraphAddChildGraphNode` cloning carries NCCL's retained user objects (if not, fall back to
cloning nodes into the parent and `cuGraphRetainUserObject` explicitly). (3) Because NCCL nodes
are rebuilt at LOAD anyway, SAVE can stub NCCL with a dummy kernel that touches the same buffers,
which lets a PP-stage-count job produce the archive.

## Runbook (single GPU first)

```bash
# both runs: identical config; Foundry built for the same torch (Foundry targets torch 2.9–2.11;
# PyTorch main changed CUDAGeneratorState to per-capture states and will not build Foundry)
export LD_PRELOAD=$FOUNDRY_SITE/foundry/libcuda_hook.so
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False   # expandable segments own VMM; conflicts with the hook
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 NCCL_GRAPH_REGISTER=0  # as already required by full_iteration

COMMON="... --cuda-graph-impl full_iteration --no-check-for-nan-in-loss-and-grad --te-rng-track \
        --cuda-graph-archive-dir /nvme/fullcg_archive"

# 1) SAVE: normal warmup (default 3 steps) + capture, then archive
python pretrain_gpt.py $COMMON --cuda-graph-persist-mode save

# 2) LOAD in a fresh process (strict makes any mismatch fatal instead of falling back)
python pretrain_gpt.py $COMMON --cuda-graph-persist-mode load --cuda-graph-persist-strict
```

What to look at after SAVE, in `<archive>/rank0/fullcg_manifest.json`:

* `stages.training.pointer_report.num_pointers` — `unmanaged > 0` means graph pointers outside
  every caching-allocator segment (NVSHMEM heap, NCCL buffers, foreign allocations); those must be
  deterministic by other means.
* `stages.training.warnings` — lists warmup-phase allocations referenced by the graph that are not
  StaticBufferLoader buffers (lazily created persistent objects; they are mapped at LOAD via
  `required_intervals`, but moving them to init is cleaner), and whether RNG is consumed.
* `stages.training.phase_offsets` — `training_first_call` must be reproduced exactly by LOAD;
  a mismatch means initialization is not deterministic (log shows both values).

Acceptance for Phase 1: LOAD-side loss bitwise equal to the native warmup+capture run on the same
data/seed; startup time; peak memory not above native.

## Design notes (why it is shaped this way)

* **Static buffers and results ride on Foundry's `output_tensors`.** Foundry rebuilds any tensor
  list it was given with `from_blob` at the recorded addresses; we hand it `result` tensors
  followed by `StaticBufferLoader` tensors and keep the container layout in a JSON schema.
* **Phases and the cursor.** `scratch` prefix absorbs non-deterministic comm-init allocations
  (same trick as Foundry's vLLM integration); the caching allocator is emptied right before the
  jump so that no block with a scratch-dependent address survives into the deterministic part
  (pointers into scratch segments are reported as class `scratch` and make LOAD refuse the
  archive); `training_first_call` marks the init/iteration boundary. Everything before it is recreated by LOAD's own init; everything between it and the
  capture window is warmup memory that LOAD never allocates, so referenced pieces are mapped
  explicitly (`required_intervals`); the capture window is replayed by Foundry's allocator events.
* **Lazily created comm buffers.** hybridep/DeepEP build their NVSHMEM buffers inside the first
  dispatch. `EagerInitRecorder` wraps those constructors in SAVE mode and records the calls
  (`eager_init_recipe.json`); every later run replays them right before the first training call so
  SAVE and LOAD allocate them at the same cursor position. An archive whose recipe was only
  recorded (not yet applied) is flagged `eager_init_pending` and refused by LOAD — SAVE is a
  two-pass procedure for MoE. LOAD calls `init_nvshmem_for_loaded_modules()` after the graph is
  rebuilt.
* **RNG.** Megatron registers `get_all_rng_states()` with the graph and requires `--te-rng-track`.
  With FP4 stochastic rounding the graph consumes RNG every step, so B1 is mandatory for
  bit-exact resume semantics, not optional.
* **Archives are per rank** by default (`rankN/`). `--cuda-graph-persist-share-across-ranks`
  keys by `pp{}_tp{}_vp{}` with a lowest-rank writer election; only valid when kernels carry no
  rank-specific arguments.

## Verified / not verified

* Verified: 15 unit tests (torch-free helpers + inspector CLI on synthetic archives);
  `py_compile` of all touched Python; shape-level syntax check of the new C++ loop. Patches apply
  cleanly to the stated commits.
* Not verified: anything on a GPU (no CUDA in the authoring environment); the C++ patches are
  written against the Foundry sources as read but were not compiled.
