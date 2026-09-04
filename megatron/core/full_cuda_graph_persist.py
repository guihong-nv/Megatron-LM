# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Foundry-backed persistence (SAVE/LOAD) of full-iteration CUDA graphs.

Overview
--------
``--cuda-graph-impl full_iteration`` captures one CUDA graph per rank for the whole
forward-backward iteration. Capturing it costs ``cuda_graph_warmup_steps`` real iterations
(JIT, autotune, lazy init) plus one CPU-side capture of the entire iteration, which for large
MoE jobs is tens of minutes. This module lets a job

* **SAVE**: run the normal warmup + capture once, then serialize the graph together with the
  execution context Foundry needs (kernel binaries, deterministic memory layout, allocator
  events) plus the Megatron-level state (static input buffers, result structure, RNG generator
  names) into an archive directory; and
* **LOAD**: in a fresh process, skip warmup and capture entirely and rebuild the executable
  graph from the archive in seconds.

Foundry (https://github.com/foundry-org/foundry) provides the LD_PRELOAD driver hook that
enforces a deterministic device memory layout and captures kernel binaries, plus the
``fdry.CUDAGraph`` save/load extension. This module only orchestrates it for Megatron.

Process phases (both modes must follow the same sequence up to "init")::

    region setup  -> comm init (scratch) -> skip_to_scratch_boundary -> init (model/optimizer/
    buffers/ckpt) -> [SAVE: warmup -> capture -> save] | [LOAD: map intervals -> load] -> steady

LOAD is fail-open (unless ``strict``) up to and including the manifest / fingerprint /
trajectory checks; once memory has been mapped or the graph rebuilt, a failure is fatal because a
native capture on top of the modified process state is not safe.

Scope / known limits (see the design discussion in the PR description):
* Phase 1 target: graphs without NCCL kernels inside (TP=1, PP=1, DP=1; EP over NVSHMEM is OK).
  NCCL collectives inside the captured region are not restorable yet (Phase 2).
* RNG rebinding requires the Foundry ``bind_generator_state`` patch. Without it, graphs that
  consume RNG (dropout, stochastic rounding) will not continue the checkpoint's RNG stream.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
import struct
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from megatron.core import full_cuda_graph_persist_utils as U

logger = logging.getLogger(__name__)

try:  # Foundry is optional; everything degrades to the regular path without it.
    import foundry as fdry
    from foundry import ops as fdry_ops

    HAVE_FOUNDRY = True
except ImportError:  # pragma: no cover - depends on the environment
    fdry = None
    fdry_ops = None
    HAVE_FOUNDRY = False


ENV_MODE = "MCORE_FULLCG_PERSIST_MODE"
ENV_ARCHIVE = "MCORE_FULLCG_ARCHIVE_DIR"
ENV_BASE_ADDR = "MCORE_FULLCG_BASE_ADDR"
ENV_REGION_SIZE = "MCORE_FULLCG_REGION_SIZE"
ENV_SCRATCH_SIZE = "MCORE_FULLCG_SCRATCH_SIZE"
ENV_STRICT = "MCORE_FULLCG_STRICT"
ENV_SHARE_RANKS = "MCORE_FULLCG_SHARE_ACROSS_RANKS"


def _parse_size(v) -> int:
    if isinstance(v, int):
        return v
    s = str(v).strip().upper()
    mult = {"B": 1, "KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30, "TB": 1 << 40}
    for suffix, m in sorted(mult.items(), key=lambda kv: -len(kv[0])):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * m)
    return int(s, 0)


def _parse_addr(v) -> int:
    if isinstance(v, int):
        return v
    return int(str(v), 0)


@dataclass
class PersistConfig:
    """Runtime configuration for full-iteration CUDA graph persistence."""

    mode: str = "none"  # none | save | load
    archive_dir: Optional[str] = None
    base_addr: int = 0x500000000000
    region_size: int = 2 << 40  # VA only; physical memory is mapped on demand
    scratch_size: int = 1 << 30  # room for non-deterministic comm-init allocations
    strict: bool = False
    share_across_ranks: bool = False
    stop_region_after_graph: bool = False

    @classmethod
    def from_args(cls, args) -> "PersistConfig":
        """Build from argparse ``args`` (TransformerConfig fields) with env-var overrides."""
        g = lambda name, default=None: getattr(args, name, default)  # noqa: E731
        mode = os.environ.get(ENV_MODE, g("cuda_graph_persist_mode", "none") or "none").lower()
        archive_dir = os.environ.get(ENV_ARCHIVE, g("cuda_graph_archive_dir"))
        base_addr = _parse_addr(
            os.environ.get(ENV_BASE_ADDR, g("cuda_graph_persist_base_addr", "0x500000000000"))
        )
        region_size = _parse_size(
            os.environ.get(ENV_REGION_SIZE, g("cuda_graph_persist_region_size", "2TB"))
        )
        scratch_size = _parse_size(
            os.environ.get(ENV_SCRATCH_SIZE, g("cuda_graph_persist_scratch_size", "1GB"))
        )
        strict = bool(int(os.environ.get(ENV_STRICT, "1" if g("cuda_graph_persist_strict") else "0")))
        share = bool(
            int(os.environ.get(ENV_SHARE_RANKS, "1" if g("cuda_graph_persist_share_across_ranks") else "0"))
        )
        if mode not in ("none", "save", "load"):
            raise ValueError(f"Invalid cuda_graph_persist_mode {mode!r}")
        if mode != "none" and not archive_dir:
            raise ValueError("cuda_graph_archive_dir must be set when cuda_graph_persist_mode != none")
        return cls(
            mode=mode,
            archive_dir=archive_dir,
            base_addr=base_addr,
            region_size=region_size,
            scratch_size=scratch_size,
            strict=strict,
            share_across_ranks=share,
        )


class PersistError(RuntimeError):
    pass


class FullCudaGraphPersistence:
    """Singleton orchestrating SAVE/LOAD. Obtain via :func:`get_persistence`."""

    def __init__(self, cfg: PersistConfig, args=None):
        self.cfg = cfg
        self.args = args
        self.mode = cfg.mode
        self.enabled = cfg.mode != "none"
        self.region_active = False
        self.phase_offsets: dict[str, int] = {}
        self.manifest: Optional[dict] = None
        self._load_attempted: dict[str, bool] = {}
        self._loaded_graph_json: dict[str, dict] = {}
        self._rank_key: Optional[str] = None
        self._hook_checked = False
        self._comm_traces: dict[str, Any] = {}
        self.timings: dict[str, dict[str, float]] = {}
        self._eager_recorder: Optional["EagerInitRecorder"] = None
        self._eager_applied = False

    def record_timing(self, stage: str, key: str, seconds: float):
        """Wall-clock bookkeeping (warmup iterations, capture, save/load) for the manifest/logs."""
        self.timings.setdefault(stage, {})[key] = round(float(seconds), 3)
        logger.info("[fullcg-timing] %s %s: %.2fs", stage, key, seconds)

    # ------------------------------------------------------------------ helpers
    def _fail(self, msg: str, exc: Optional[BaseException] = None):
        """Strict -> raise; otherwise log and disable persistence for this process."""
        full = f"[fullcg-persist] {msg}"
        if exc is not None:
            full += f" ({type(exc).__name__}: {exc})"
        if self.cfg.strict:
            raise PersistError(full) from exc
        logger.warning(full + " -- falling back to regular warmup + capture.")
        self.enabled = False

    def _rank(self) -> int:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        r = getattr(self.args, "rank", None)
        if r is None:
            r = int(os.environ.get("RANK", "0"))
        return int(r)

    def rank_key(self) -> str:
        """Archive sub-directory key. Per-rank by default; PP/TP-role only when sharing."""
        if self._rank_key is not None:
            return self._rank_key
        if self.cfg.share_across_ranks:
            try:
                from megatron.core import parallel_state as ps

                pp = ps.get_pipeline_model_parallel_rank()
                tp = ps.get_tensor_model_parallel_rank()
                vp = ps.get_virtual_pipeline_model_parallel_rank() or 0
                self._rank_key = f"pp{pp}_tp{tp}_vp{vp}"
            except Exception:  # model parallel not initialized yet
                self._rank_key = f"rank{self._rank()}"
        else:
            self._rank_key = f"rank{self._rank()}"
        return self._rank_key

    def archive_dir(self) -> str:
        return os.path.join(self.cfg.archive_dir, self.rank_key())

    def cursor(self) -> int:
        return int(fdry_ops.get_current_alloc_offset()) if (HAVE_FOUNDRY and self.region_active) else -1

    def _all_ranks_agree(self, ok: bool) -> bool:
        if not torch.distributed.is_initialized():
            return ok
        t = getattr(self, "_flag", None)
        if t is None:  # before after_comm_init (only reachable in odd init orders)
            t = torch.zeros(1, dtype=torch.int32, device="cuda")
        t.fill_(1 if ok else 0)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
        return bool(t.item())

    def _check_hook_preloaded(self):
        if self._hook_checked:
            return
        self._hook_checked = True
        preload = os.environ.get("LD_PRELOAD", "")
        if "cuda_hook" not in preload and "foundry" not in preload:
            self._fail(
                "LD_PRELOAD does not contain Foundry's libcuda_hook.so; the deterministic "
                "allocation region cannot work. Launch every rank with LD_PRELOAD=<foundry>/libcuda_hook.so"
            )

    # -------------------------------------------------------------- lifecycle
    def early_init(self):
        """Call right after ``torch.cuda.set_device`` and before any device allocation."""
        if not self.enabled:
            return
        if not HAVE_FOUNDRY:
            self._fail("foundry python package not importable")
            return
        self._check_hook_preloaded()
        if not self.enabled:
            return
        try:
            torch.cuda.synchronize()  # force primary context creation on this device
            if self.mode == "load":
                # The hook constructor only sees FOUNDRY_MODE from the launcher environment;
                # this is the in-process equivalent (skip fatbin extraction work in LOAD).
                fdry_ops.set_skip_fatbin_processing(True)
            fdry_ops.set_allocation_region(self.cfg.base_addr, self.cfg.region_size)
            self.region_active = True
            # Prove the LD_PRELOAD hook is really intercepting this process: the first device
            # allocation must land inside the region. (Identical in SAVE and LOAD, so it is
            # part of the deterministic trajectory.)
            probe = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
            ptr = int(probe.data_ptr())
            del probe
            if not (self.cfg.base_addr <= ptr < self.cfg.base_addr + self.cfg.region_size):
                self.region_active = False
                self._fail(
                    f"probe allocation at 0x{ptr:x} is outside the allocation region; the Foundry "
                    "hook is not active in this process (LD_PRELOAD missing or wrong library)"
                )
                return
            self.phase_offsets["region_start"] = self.cursor()
            logger.info(
                "[fullcg-persist] allocation region base=0x%x size=%d mode=%s",
                self.cfg.base_addr,
                self.cfg.region_size,
                self.mode,
            )
        except Exception as e:  # pragma: no cover
            self._fail("failed to set up Foundry allocation region", e)

    def after_comm_init(self):
        """Call after torch.distributed + model-parallel groups exist, before model construction.

        Eagerly initializes every process group (their NCCL buffers land in the scratch prefix),
        then moves the cursor to a fixed boundary so that everything allocated afterwards is at
        the same offset in SAVE and LOAD regardless of how much NCCL allocated.
        """
        if not self.enabled or not self.region_active:
            return
        try:
            warm_up_process_groups()
            if self.mode == "load":
                # Kernel binaries used by the graph, keyed by (content hash, mangled name). Done
                # here (rank keys need parallel_state) and before the cursor jump, so whatever
                # the module loader allocates is absorbed by the scratch prefix.
                t0 = time.perf_counter()
                fdry_ops.load_cuda_modules_and_libraries(self.archive_dir())
                logger.info(
                    "[fullcg-persist] loaded kernel binaries from %s in %.2fs",
                    self.archive_dir(),
                    time.perf_counter() - t0,
                )
            # Release every cached caching-allocator segment that lives in the scratch prefix
            # (the hook probe, warm-up buffers, barrier tensors). Their addresses depend on how
            # much NCCL allocated and therefore differ between SAVE and LOAD; if PyTorch kept them
            # cached it would later place model/optimizer tensors in them.
            import gc

            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            cur = self.cursor()
            if cur > self.cfg.scratch_size:
                self._fail(
                    f"comm init consumed {cur} bytes of the region but scratch_size is only "
                    f"{self.cfg.scratch_size}; raise --cuda-graph-persist-scratch-size"
                )
                return
            fdry_ops.set_current_alloc_offset(self.cfg.scratch_size)
            self.phase_offsets["scratch_boundary"] = self.cursor()
            logger.info(
                "[fullcg-persist] comm scratch used %d bytes, cursor moved to 0x%x", cur, self.cursor()
            )
            # Deterministic (post-boundary) scratch tensor for collective agreement checks, so
            # that LOAD never allocates between the trajectory check and the interval mapping.
            self._flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        except Exception as e:  # pragma: no cover
            self._fail("after_comm_init failed", e)
        # Fail-open must be collective: a rank that disabled persistence must not be paired with
        # ranks that later call save()/barrier() or try_load().
        self._sync_enabled()
        self.start_eager_recorder()

    def _sync_enabled(self):
        if not torch.distributed.is_initialized():
            return
        ok = self._all_ranks_agree(self.enabled)
        if self.enabled and not ok:
            logger.warning("[fullcg-persist] persistence disabled on another rank; disabling here too")
            self.enabled = False

    # ---------------------------------------------------------- lazily created comm buffers
    def apply_eager_init(self, stage: str):
        """Create lazily-initialized communication buffers at a fixed trajectory point.

        hybridep / DeepEP allocate their NVSHMEM buffers inside the *first dispatch*, i.e. in the
        middle of warmup iteration 0. A LOAD run never executes that forward, and even if it
        re-created the buffers later they would land at a different cursor position (NVSHMEM's heap
        reservation goes through the hooked cuMemAddressReserve). The only robust fix is to create
        them at the same point in every run: here, right before the first training call.

        SAVE pass 1 records the constructor calls it observes (``eager_init_recipe.json``) and flags
        the archive ``eager_init_pending`` if any happened after the first call; SAVE pass 2 (and
        every LOAD) replays the recipe here, so the lazily-initialized objects become part of the
        deterministic init phase. Dense models record nothing and need a single pass.
        """
        if not self.enabled or self._eager_applied or stage != "training":
            return
        self._eager_applied = True
        recipe_path = os.path.join(self.archive_dir(), EAGER_INIT_RECIPE)
        if not os.path.exists(recipe_path):
            return
        try:
            with open(recipe_path) as f:
                recipe = json.load(f)
            n = apply_eager_init_recipe(recipe)
            logger.info("[fullcg-persist] applied eager-init recipe (%d calls) before first call", n)
        except Exception as e:  # noqa: BLE001
            self._fail("applying eager-init recipe failed", e)

    def start_eager_recorder(self):
        if self.enabled and self.mode == "save" and self._eager_recorder is None:
            self._eager_recorder = EagerInitRecorder(self)
            self._eager_recorder.install()

    def mark_stage_first_call(self, stage: str):
        """Record the cursor the first time the wrapper is called for ``stage``.

        For 'training' this is the init/iteration boundary ("init_end"): everything allocated
        before it is recreated by LOAD's own initialization; everything between it and the
        capture window is warmup-phase memory that LOAD never allocates by itself.
        """
        if not self.enabled or not self.region_active:
            return
        key = f"{stage}_first_call"
        if key in self.phase_offsets:
            return
        self.phase_offsets[key] = self.cursor()
        if stage == "training":
            self.phase_offsets["init_end"] = self.phase_offsets[key]
        logger.info("[fullcg-persist] %s cursor offset 0x%x", key, self.phase_offsets[key])

    def on_graph_ready(self, stage: str):
        """Called after the graph for ``stage`` is captured+saved or loaded."""
        # Only leave the deterministic region once no further graph will be captured/loaded
        # in it. The validation graph is captured lazily at the first eval, so by default we
        # keep the region active for the whole process (steady-state allocations are rare and
        # the VMM slow path costs microseconds).
        if stage == "training" and self.cfg.stop_region_after_graph:
            self.stop_region()

    def stop_region(self):
        if self.region_active and HAVE_FOUNDRY:
            fdry_ops.stop_allocation_region()
            self.region_active = False
            logger.info("[fullcg-persist] allocation region stopped (steady state)")

    # ------------------------------------------------------------ capture side
    def new_graph(self):
        """Graph object to capture into: Foundry's in SAVE mode, torch's otherwise."""
        if self.enabled and self.mode == "save":
            return fdry.CUDAGraph()
        return torch.cuda.CUDAGraph()

    def capture_context(self, cuda_graph, stream, pool, capture_error_mode="thread_local"):
        """Context manager equivalent to ``torch.cuda.graph`` for the chosen graph type."""
        if self.enabled and self.mode == "save":
            return fdry.graph(cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode)
        return torch.cuda.graph(cuda_graph, stream=stream, pool=pool, capture_error_mode=capture_error_mode)

    def _is_archive_writer(self) -> bool:
        """With share_across_ranks several ranks map to one key; only the lowest rank writes."""
        if not self.cfg.share_across_ranks or not torch.distributed.is_initialized():
            return True
        world = torch.distributed.get_world_size()
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, (self.rank_key(), self._rank()))
        owner = min(r for k, r in gathered if k == self.rank_key())
        return owner == self._rank()

    def comm_trace(self, stage: str):
        """Context manager recording torch.distributed calls during capture (SAVE mode).

        Returns a no-op context when disabled. The trace is written next to the graph by save()
        and is the input for Phase 2 (NCCL placeholder re-capture); `tools/fullcg_graph_inspect.py
        comm` compares it with the NCCL kernel count in the graph.
        """
        import contextlib

        if not (self.enabled and self.mode == "save") or os.environ.get("MCORE_FULLCG_COMM_TRACE", "1") == "0":
            return contextlib.nullcontext()
        from megatron.core.full_cuda_graph_comm_trace import CommTraceRecorder

        rec = CommTraceRecorder()
        self._comm_traces[stage] = rec
        return rec

    def save(self, stage: str, cuda_graph, result, static_buffers, generator_states: dict):
        """Serialize graph + Megatron state for ``stage`` ('training' | 'validation')."""
        if not (self.enabled and self.mode == "save"):
            return
        try:
            if self._is_archive_writer():
                self._save_impl(stage, cuda_graph, result, static_buffers, generator_states)
            else:
                logger.info("[fullcg-persist] rank %d shares key %s; not writing", self._rank(), self.rank_key())
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception as e:
            self._fail(f"SAVE failed for stage {stage}", e)

    def _save_impl(self, stage, cuda_graph, result, static_buffers, generator_states):
        adir = self.archive_dir()
        os.makedirs(adir, exist_ok=True)
        names = U.stage_archive_names(stage)
        is_tensor = lambda x: isinstance(x, torch.Tensor)  # noqa: E731

        # Flat tensor list handed to Foundry = result tensors followed by static input buffers.
        # Foundry records (data_ptr, sizes, strides, dtype) and rebuilds them with from_blob at
        # LOAD, so both the graph outputs and the static inputs come back at the same addresses.
        result_schema, tensors = U.flatten_struct(result, is_tensor)
        static_schema, tensors = U.flatten_struct(static_buffers, is_tensor, tensors)
        flat = [t.detach() if t.requires_grad else t for t in tensors]
        for t in flat:
            if not t.is_cuda:
                raise PersistError("non-CUDA tensor in graph result/static buffers")

        # Optional: name generator states so LOAD can rebind them (needs Foundry patch B1).
        gen_names = []
        for name, gen in generator_states.items():
            gen_names.append(name)
            if hasattr(cuda_graph, "set_generator_name"):
                cuda_graph.set_generator_name(gen, name)

        t0 = time.perf_counter()
        graph_json = os.path.join(adir, names["graph_json"])
        cuda_graph.save(graph_json, output_tensors=flat)
        logger.info("[fullcg-persist] graph JSON/binary written in %.2fs", time.perf_counter() - t0)

        # Kernel binaries used by the graph(s) captured so far.
        fdry_ops.pack_fatbins_to_folder(adir)
        fdry_ops.set_pack_fatbins_on_exit(False)

        # Pointer provenance: which allocations does the graph reference, and which phase made
        # them? Warmup-phase segments must be mapped explicitly at LOAD.
        with open(graph_json) as f:
            gj = json.load(f)
        capture_start = int(gj["allocator_events"]["start_base_addr"])
        first_call_off = self.phase_offsets.get(f"{stage}_first_call", self.phase_offsets.get("init_end", 0))
        init_end = self.cfg.base_addr + first_call_off
        lo, hi = self.cfg.base_addr, self.cfg.base_addr + self.cfg.region_size
        pointers = set(U.extract_candidate_pointers(gj, lo, hi))
        pointers.update(int(t.data_ptr()) for t in flat)
        # RNG extragraph tensors (seed/offset read by dropout / stochastic-rounding kernels) must
        # be mapped at LOAD too; the kernel-arg scan normally finds them, this makes it explicit.
        for g in gj.get("generators", []):
            if int(g.get("wholegraph_increment", 0)) > 0:
                for key in ("seed_extragraph_ptr", "offset_extragraph_ptr"):
                    if int(g.get(key, 0)):
                        pointers.add(int(g[key]))
        segments = U.segments_from_snapshot(torch.cuda.memory_snapshot(), device=torch.cuda.current_device())
        scratch_end = self.cfg.base_addr + self.phase_offsets.get("scratch_boundary", 0)
        classified = U.classify_pointers(pointers, segments, init_end, capture_start, scratch_end)
        required = U.required_intervals(classified)
        report = U.summarize_pointer_report(classified)

        warnings = []
        if classified["scratch"]:
            msg = (
                f"{len(classified['scratch'])} graph pointers refer to comm-scratch segments whose "
                "addresses are not reproducible at LOAD (a caching-allocator block from before the "
                "scratch jump was reused). This archive cannot be restored correctly."
            )
            if self.cfg.strict:
                raise PersistError(msg)
            warnings.append(msg)
        if classified["unmanaged"]:
            warnings.append(
                f"{len(classified['unmanaged'])} graph pointers are outside every caching-allocator "
                "segment (NVSHMEM heap / NCCL buffers / foreign allocations); they must be "
                "deterministic by other means. See pointer_report.unmanaged_pointers."
            )
        static_ptrs = {int(t.data_ptr()) for t in flat[U.count_tensors(result_schema) :]}
        extra_warmup = [
            hex(p) for p, _ in classified["warmup"] if p not in static_ptrs
        ]
        if extra_warmup:
            warnings.append(
                f"{len(extra_warmup)} graph pointers refer to warmup-phase allocations that are "
                "not static input buffers (lazily created persistent objects). They are mapped "
                "at LOAD via required_intervals, but consider allocating them during init."
            )
        for w in warnings:
            logger.warning("[fullcg-persist] %s", w)

        generators = gj.get("generators", [])
        rng_consumed = any(int(g.get("wholegraph_increment", 0)) > 0 for g in generators)
        if rng_consumed and not hasattr(cuda_graph, "set_generator_name"):
            warnings.append(
                "graph consumes RNG (dropout/stochastic rounding) but this Foundry build lacks "
                "generator naming/binding; LOAD cannot continue the checkpoint's RNG stream."
            )
            logger.warning("[fullcg-persist] %s", warnings[-1])

        comm_info = {}
        rec = self._comm_traces.pop(stage, None)
        if rec is not None:
            trace_path = os.path.join(adir, f"comm_trace_{stage}.json")
            rec.save(trace_path)
            comm_info = dict(rec.summary(), path=os.path.basename(trace_path))
        nccl_nodes = [
            n["id"]
            for n in gj.get("nodes", [])
            if n.get("type") == "KernelNode" and "nccl" in (n.get("params", {}).get("function_name", "") or "").lower()
        ]
        comm_info["nccl_kernel_nodes"] = len(nccl_nodes)
        if nccl_nodes:
            warnings.append(
                f"{len(nccl_nodes)} NCCL kernel nodes inside the graph: Phase-1 LOAD will rebuild them "
                "verbatim, which is NOT valid (NCCL plans/buffers are per-process). Use this archive "
                "for diagnostics only until Phase-2 placeholder re-capture lands."
            )
            logger.warning("[fullcg-persist] %s", warnings[-1])

        manifest = U.read_manifest(adir)
        fp_fields = self.fingerprint_fields()
        fp = U.compute_fingerprint(fp_fields)
        if manifest is None or manifest.get("fingerprint") != fp:
            manifest = U.build_manifest(fp, fp_fields, self.cfg.base_addr, self.cfg.region_size, self.rank_key(), self.env_info())
        U.add_stage_to_manifest(
            manifest,
            stage,
            result_schema=result_schema,
            static_buffers_schema=static_schema,
            num_tensors=len(flat),
            generator_names=gen_names,
            phase_offsets=dict(self.phase_offsets, capture_start_abs=capture_start, rng_consumed=rng_consumed),
            required=required,
            pointer_report=report,
            warnings=warnings,
        )
        manifest["stages"][stage]["comm"] = comm_info
        manifest["stages"][stage]["timing"] = dict(self.timings.get(stage, {}))
        pending = False
        if self._eager_recorder is not None and self._eager_recorder.calls:
            first_call_off = self.phase_offsets.get("training_first_call", 0)
            pending = any(c["cursor_offset"] > first_call_off for c in self._eager_recorder.calls)
            with open(os.path.join(adir, EAGER_INIT_RECIPE), "w") as f:
                json.dump({"version": 1, "calls": self._eager_recorder.calls}, f, indent=1)
            if pending:
                warnings.append(
                    f"{len(self._eager_recorder.calls)} lazily-initialized comm buffer(s) were created "
                    "during warmup (after the first training call). Their recipe was recorded; RE-RUN "
                    "SAVE with the same command so they are created before the first call. LOAD "
                    "refuses this archive (eager_init_pending)."
                )
                logger.warning("[fullcg-persist] %s", warnings[-1])
                manifest["stages"][stage]["warnings"] = warnings
        manifest["eager_init_pending"] = pending
        U.write_manifest(adir, manifest)
        logger.info(
            "[fullcg-persist] SAVE %s: %d tensors, %d required warmup intervals, report=%s",
            stage,
            len(flat),
            len(required),
            json.dumps(report["num_pointers"]),
        )

    # --------------------------------------------------------------- load side
    def try_load(self, stage: str, generator_states: dict):
        """Rebuild the graph for ``stage``. Returns (graph, result, static_buffers) or None.

        Must run before ``StaticBufferLoader`` allocates anything for this stage (i.e. at the
        top of ``FullCudaGraphWrapper.__call__``) and only once per stage.
        """
        if not (self.enabled and self.mode == "load") or self._load_attempted.get(stage):
            return None
        self._load_attempted[stage] = True
        # Phase 1: pure checks (manifest, fingerprint, trajectory). Nothing mutated yet, so
        # fail-open is still safe. Agreement across ranks decides for everyone.
        ok, st, err = True, None, None
        try:
            st = self._load_check(stage)
        except Exception as e:  # noqa: BLE001
            ok, err = False, e
        if not self._all_ranks_agree(ok):
            if ok:
                logger.warning("[fullcg-persist] another rank failed the LOAD checks for %s; falling back on all ranks", stage)
            else:
                self._fail(f"LOAD checks failed for stage {stage}", err)
            self.enabled = False
            return None
        # Phase 2: mutating steps (interval mapping, allocator replay, graph build). A failure
        # here leaves mapped memory the caching allocator does not know about, so falling back
        # to a native capture is not safe -> always fatal.
        try:
            return self._load_apply(stage, st, generator_states)
        except Exception as e:  # noqa: BLE001
            raise PersistError(
                f"[fullcg-persist] LOAD of stage {stage} failed after the process state was "
                f"modified; cannot fall back. ({type(e).__name__}: {e})"
            ) from e

    def _load_check(self, stage) -> dict:
        adir = self.archive_dir()
        manifest = U.read_manifest(adir)
        if manifest is None:
            raise PersistError(f"no manifest in {adir}")
        if manifest.get("version") != U.MANIFEST_VERSION:
            raise PersistError("manifest version mismatch")
        fp_fields = self.fingerprint_fields()
        fp = U.compute_fingerprint(fp_fields)
        if manifest["fingerprint"] != fp:
            diffs = U.diff_fingerprint_fields(manifest.get("fingerprint_fields", {}), fp_fields)
            raise PersistError("fingerprint mismatch: " + "; ".join(diffs))
        region = manifest["region"]
        if int(region["base"]) != self.cfg.base_addr or int(region["size"]) != self.cfg.region_size:
            raise PersistError("allocation region base/size differ from the archive")
        if stage not in manifest["stages"]:
            raise PersistError(f"stage {stage} not in archive")
        st = manifest["stages"][stage]
        if manifest.get("eager_init_pending"):
            raise PersistError(
                "archive was produced by a SAVE pass that created comm buffers lazily during warmup; "
                "re-run SAVE (the recorded eager_init_recipe.json is applied automatically) before LOAD"
            )
        if int(((st.get("pointer_report") or {}).get("num_pointers") or {}).get("scratch", 0)) > 0:
            raise PersistError("archive graph references comm-scratch memory (see SAVE warnings)")
        if int((st.get("comm") or {}).get("nccl_kernel_nodes", 0)) > 0 and not os.environ.get(
            "MCORE_FULLCG_ALLOW_NCCL_NODES"
        ):
            raise PersistError(
                "archive graph contains NCCL kernel nodes; restoring them verbatim would replay "
                "another process's communicator state. Phase-2 (placeholder re-capture) required. "
                "Set MCORE_FULLCG_ALLOW_NCCL_NODES=1 to override for experiments."
            )

        # Trajectory check: everything before this stage's first call must have consumed exactly
        # as much of the region as it did at SAVE, otherwise addresses differ.
        saved_first = int(st["phase_offsets"].get(f"{stage}_first_call", -1))
        cur = self.cursor()
        if saved_first >= 0 and cur != saved_first:
            raise PersistError(
                f"cursor at {stage} first call is 0x{cur:x} but SAVE recorded 0x{saved_first:x}; "
                "allocation trajectory is not deterministic between SAVE and LOAD"
            )
        saved_scratch = int(st["phase_offsets"].get("scratch_boundary", -1))
        cur_scratch = int(self.phase_offsets.get("scratch_boundary", -1))
        if saved_scratch >= 0 and saved_scratch != cur_scratch:
            raise PersistError(f"scratch boundary differs: SAVE 0x{saved_scratch:x}, LOAD 0x{cur_scratch:x}")
        for name in ("graph_json", "graph_bin"):
            if not os.path.exists(os.path.join(adir, st[name])):
                raise PersistError(f"missing {st[name]} in {adir}")
        return st

    def _load_apply(self, stage, st, generator_states):
        adir = self.archive_dir()
        # 2. Map warmup-phase segments the graph references (static input buffers etc.).
        t0 = time.perf_counter()
        self._map_intervals([(int(a), int(s)) for a, s in st["required_intervals"]])

        # 3. Foundry: replays capture-window allocator events, reloads kernels by (hash, name),
        #    rebuilds the CUgraph via driver APIs, instantiates, and recreates output tensors.
        graph_json = os.path.join(adir, st["graph_json"])
        pool = None
        loaded = fdry.CUDAGraph.load(graph_json, pool)
        # Foundry returns the bare graph when the archive carried no output tensors.
        if isinstance(loaded, tuple):
            graph, outputs = loaded
        else:
            graph, outputs = loaded, []
        outputs = list(outputs or [])
        if len(outputs) != int(st["num_tensors"]):
            raise PersistError(f"archive has {st['num_tensors']} tensors, load returned {len(outputs)}")
        n_res = U.count_tensors(st["result_schema"])
        result = U.unflatten_struct(st["result_schema"], outputs[:n_res])
        static_buffers = U.unflatten_struct(st["static_buffers_schema"], outputs[n_res:])
        logger.info(
            "[fullcg-persist] LOAD %s: graph rebuilt in %.2fs (%d tensors)",
            stage,
            time.perf_counter() - t0,
            len(outputs),
        )

        # NVSHMEM-backed kernels (hybridep / DeepEP) need their module-side NVSHMEM state set up in
        # this process; the runtime itself was initialized by the eager-init recipe above.
        try:
            n_nvshmem = fdry_ops.init_nvshmem_for_loaded_modules()
            if n_nvshmem:
                logger.info("[fullcg-persist] initialized NVSHMEM for %d loaded modules", n_nvshmem)
        except Exception as e:  # noqa: BLE001
            logger.warning("[fullcg-persist] init_nvshmem_for_loaded_modules failed: %s", e)

        # 4. RNG: rebind the graph's generator slots to the live (checkpoint-restored) states.
        self._bind_generators(graph, graph_json, st, generator_states)
        with open(graph_json) as f:
            self._loaded_graph_json[stage] = {"generators": json.load(f).get("generators", [])}
        return graph, result, static_buffers

    def _map_intervals(self, intervals):
        """Physically map ``[(addr, size)]`` inside the region (warmup-phase segments)."""
        if not intervals:
            return
        if hasattr(fdry_ops, "preallocate_intervals"):  # Foundry patch A3
            ok = fdry_ops.preallocate_intervals([(int(a), int(s)) for a, s in intervals])
            if not ok:
                raise PersistError("preallocate_intervals failed")
            return
        # Fallback without the patch: walk the cursor to each interval and allocate through the
        # hooked driver entry point (LD_PRELOAD intercepts ctypes calls too via its dlsym hook).
        lib = ctypes.CDLL("libcuda.so.1")
        cu_alloc = lib.cuMemAlloc_v2
        cu_alloc.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t]
        cu_alloc.restype = ctypes.c_int
        for addr, size in sorted(intervals):
            off = addr - self.cfg.base_addr
            if off < self.cursor():
                raise PersistError(f"interval 0x{addr:x} is below the current cursor")
            fdry_ops.set_current_alloc_offset(off)
            p = ctypes.c_ulonglong(0)
            rc = cu_alloc(ctypes.byref(p), ctypes.c_size_t(size))
            if rc != 0 or p.value != addr:
                raise PersistError(f"mapping interval 0x{addr:x}+{size} failed (rc={rc}, got 0x{p.value:x})")

    def _bind_generators(self, graph, graph_json, st, generator_states):
        names = st.get("generator_names", [])
        if hasattr(graph, "bind_generator_state"):
            for name in names:
                gen = generator_states.get(name)
                if gen is None:
                    raise PersistError(f"generator {name!r} present at SAVE is missing at LOAD")
                graph.bind_generator_state(name, gen)
            logger.info("[fullcg-persist] rebound %d generator states", len(names))
            return
        if st["phase_offsets"].get("rng_consumed"):
            msg = (
                "graph consumes RNG but this Foundry build has no bind_generator_state(); the "
                "restored graph will NOT continue the checkpoint's RNG stream"
            )
            if self.cfg.strict:
                raise PersistError(msg)
            logger.warning("[fullcg-persist] %s", msg)

    # ------------------------------------------------------------ fingerprint
    def fingerprint_fields(self) -> dict:
        fields = {}
        if self.args is not None:
            fields.update(U.select_fingerprint_fields(vars(self.args)))
        fields.update(self.env_info())
        try:
            from megatron.core import parallel_state as ps

            fields["pp_rank"] = ps.get_pipeline_model_parallel_rank()
            fields["is_pp_first"] = ps.is_pipeline_first_stage()
            fields["is_pp_last"] = ps.is_pipeline_last_stage()
        except Exception:
            pass
        fields["region_base"] = self.cfg.base_addr
        fields["region_size"] = self.cfg.region_size
        fields["scratch_size"] = self.cfg.scratch_size
        return fields

    @staticmethod
    def env_info() -> dict:
        info = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        }
        try:
            info["nccl"] = ".".join(str(x) for x in torch.cuda.nccl.version())
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(torch.cuda.current_device())
                info["gpu"] = f"{p.name} sm{p.major}{p.minor}"
        except Exception:
            pass
        for mod in ("transformer_engine", "megatron.core"):
            try:
                m = __import__(mod, fromlist=["__version__"])
                info[mod] = getattr(m, "__version__", "unknown")
            except Exception:
                pass
        return info


# ----------------------------------------------------------------- eager init of comm buffers

EAGER_INIT_RECIPE = "eager_init_recipe.json"

# (module path, function name) of constructors that lazily create NVSHMEM/IPC comm buffers on
# the first dispatch. Extend when a new backend appears (the count mismatch shows up as an
# `eager_init_pending` archive or a stale-pointer LOAD).
EAGER_INIT_TARGETS = (
    ("megatron.core.transformer.moe.fused_a2a", "init_hybrid_ep_buffer"),
    ("megatron.core.transformer.moe.fused_a2a", "get_buffer"),
)


def _group_getter_name(group) -> Optional[str]:
    """Name of the parallel_state getter whose group has the same ranks (JSON-able group id)."""
    try:
        from megatron.core import parallel_state as ps

        ranks = list(torch.distributed.get_process_group_ranks(group))
        for name in _GROUP_GETTERS:
            getter = getattr(ps, name, None)
            if getter is None:
                continue
            try:
                g = getter()
            except Exception:  # noqa: BLE001
                continue
            if g is None or isinstance(g, (list, tuple)):
                continue
            if list(torch.distributed.get_process_group_ranks(g)) == ranks:
                return name
    except Exception:  # noqa: BLE001
        pass
    return None


def _jsonable_arg(v):
    if isinstance(v, torch.distributed.ProcessGroup):
        name = _group_getter_name(v)
        if name is None:
            raise PersistError("comm buffer created on a process group not reachable from parallel_state")
        return {"__group__": name}
    if isinstance(v, torch.dtype):
        return {"__dtype__": str(v)}
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise PersistError(f"cannot record eager-init argument of type {type(v)}")


def _restore_arg(v):
    if isinstance(v, dict) and "__group__" in v:
        from megatron.core import parallel_state as ps

        return getattr(ps, v["__group__"])()
    if isinstance(v, dict) and "__dtype__" in v:
        return getattr(torch, v["__dtype__"].split(".")[-1])
    return v


class EagerInitRecorder:
    """Wrap lazily-called comm-buffer constructors and record their arguments (SAVE mode)."""

    def __init__(self, persist: "FullCudaGraphPersistence"):
        self.persist = persist
        self.calls: list[dict] = []
        self._installed: list[tuple[Any, str, Any]] = []

    def install(self):
        import importlib
        import inspect

        for mod_name, fn_name in EAGER_INIT_TARGETS:
            try:
                mod = importlib.import_module(mod_name)
            except Exception:  # noqa: BLE001
                continue
            orig = getattr(mod, fn_name, None)
            if orig is None:
                continue
            sig = inspect.signature(orig)
            recorder = self

            def make_wrapper(orig=orig, sig=sig, mod_name=mod_name, fn_name=fn_name):
                import functools

                @functools.wraps(orig)
                def wrapper(*args, **kwargs):
                    try:
                        bound = sig.bind(*args, **kwargs)
                        bound.apply_defaults()
                        rec = {
                            "module": mod_name,
                            "function": fn_name,
                            "kwargs": {k: _jsonable_arg(v) for k, v in bound.arguments.items()},
                            "cursor_offset": recorder.persist.cursor(),
                        }
                        recorder.calls.append(rec)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("[fullcg-persist] could not record %s.%s: %s", mod_name, fn_name, e)
                    return orig(*args, **kwargs)

                return wrapper

            setattr(mod, fn_name, make_wrapper())
            self._installed.append((mod, fn_name, orig))

    def uninstall(self):
        for mod, fn_name, orig in self._installed:
            setattr(mod, fn_name, orig)
        self._installed.clear()


def apply_eager_init_recipe(recipe: dict) -> int:
    """Replay recorded constructor calls (idempotent: constructors keep module-level singletons)."""
    import importlib

    n = 0
    for call in recipe.get("calls", []):
        mod = importlib.import_module(call["module"])
        fn = getattr(mod, call["function"])
        # unwrap a still-installed recorder wrapper so the replay itself is not re-recorded
        fn = getattr(fn, "__wrapped__", fn)
        kwargs = {k: _restore_arg(v) for k, v in call["kwargs"].items()}
        fn(**kwargs)
        n += 1
    return n


# ------------------------------------------------------------------------------ module API

_PERSISTENCE: Optional[FullCudaGraphPersistence] = None

ENV_LOSS_DUMP = "MCORE_FULLCG_LOSS_DUMP"
_LOSS_DUMP_STATE: dict[str, int] = {}


def dump_result_if_requested(stage: str, result) -> None:
    """Test aid: append the exact float bits of every scalar in the iteration result to a file.

    Enabled by ``MCORE_FULLCG_LOSS_DUMP=/path/prefix`` (one JSONL file per rank:
    ``<prefix>.rank<N>.jsonl``). Works with or without persistence so the native warmup+capture
    run and the LOAD run can be compared bitwise with ``tools/fullcg_compare_loss.py``. The
    Megatron log prints losses with 6 decimals, which is not enough for a bitwise check.
    Costs one device sync per iteration; never enable in production.
    """
    prefix = os.environ.get(ENV_LOSS_DUMP)
    if not prefix or result is None:
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    path = f"{prefix}.rank{rank}.jsonl"
    local_it = _LOSS_DUMP_STATE.get(stage, 0)
    _LOSS_DUMP_STATE[stage] = local_it + 1
    it = local_it
    try:  # global iteration number (megatron.training sets args.curr_iteration in the train loop)
        from megatron.training.global_vars import get_args

        g = getattr(get_args(), "curr_iteration", None)
        if g is not None and stage == "training":
            it = int(g)
    except Exception:  # noqa: BLE001 - core may run without megatron.training
        pass
    is_tensor = lambda x: isinstance(x, torch.Tensor)  # noqa: E731
    schema, tensors = U.flatten_struct(result, is_tensor)
    torch.cuda.synchronize()
    rec = {"stage": stage, "iteration": it, "local_iteration": local_it, "values": []}
    for i, t in enumerate(tensors):
        flat = t.detach().reshape(-1)
        if flat.numel() > 64:  # only scalars/small vectors (losses, token counts)
            continue
        vals = flat.to(torch.float64).cpu().tolist() if flat.is_floating_point() else flat.cpu().tolist()
        bits = [struct.pack("<d", float(v)).hex() if isinstance(v, float) else int(v) for v in vals]
        rec["values"].append({"tensor": i, "dtype": str(t.dtype), "shape": list(t.shape), "bits": bits})
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def init_persistence(args) -> FullCudaGraphPersistence:
    """Create the singleton from parsed args (idempotent)."""
    global _PERSISTENCE
    if _PERSISTENCE is None:
        _PERSISTENCE = FullCudaGraphPersistence(PersistConfig.from_args(args), args)
    return _PERSISTENCE


def get_persistence() -> Optional[FullCudaGraphPersistence]:
    """Return the singleton if persistence is enabled, else None."""
    if _PERSISTENCE is not None and _PERSISTENCE.enabled:
        return _PERSISTENCE
    return None


def named_generator_states() -> dict[str, Any]:
    """{name: generator} for everything FullCudaGraphWrapper registers with the graph."""
    states: dict[str, Any] = {}
    try:
        dev = torch.cuda.current_device()
        states["default"] = torch.cuda.default_generators[dev]
    except Exception:
        pass
    try:
        from megatron.core.tensor_parallel.random import get_all_rng_states

        for name, st in get_all_rng_states().items():
            states[str(name)] = st
    except Exception:
        pass
    return states


_GROUP_GETTERS = (
    "get_model_parallel_group",
    "get_tensor_model_parallel_group",
    "get_pipeline_model_parallel_group",
    "get_data_parallel_group",
    "get_context_parallel_group",
    "get_embedding_group",
    "get_position_embedding_group",
    "get_tensor_and_data_parallel_group",
    "get_tensor_and_context_parallel_group",
    "get_expert_model_parallel_group",
    "get_expert_tensor_parallel_group",
    "get_expert_tensor_and_model_parallel_group",
    "get_expert_tensor_model_pipeline_parallel_group",
    "get_expert_data_parallel_group",
    "get_intra_distributed_optimizer_instance_group",
    "get_inter_distributed_optimizer_instance_group",
)


def warm_up_process_groups() -> int:
    """Eagerly initialize NCCL communicators of all model-parallel groups in a fixed order.

    NCCL communicators are otherwise created on first use, which happens somewhere inside the
    warmup iterations at SAVE time and would shift every later allocation relative to LOAD.
    Returns the number of groups touched.
    """
    if not torch.distributed.is_initialized():
        return 0
    from megatron.core import parallel_state as ps

    count = 0
    buf = torch.ones(1, device="cuda", dtype=torch.float32)
    for name in _GROUP_GETTERS:
        getter = getattr(ps, name, None)
        if getter is None:
            continue
        try:
            group = getter()
        except Exception:
            continue
        if group is None:
            continue
        groups = group if isinstance(group, (list, tuple)) else [group]
        for g in groups:
            try:
                torch.distributed.all_reduce(buf, group=g)
                count += 1
            except Exception as e:  # noqa: BLE001
                logger.debug("[fullcg-persist] warm-up of %s failed: %s", name, e)
    torch.distributed.all_reduce(buf)  # default group
    torch.cuda.synchronize()
    logger.info("[fullcg-persist] eagerly initialized %d process groups", count)
    return count + 1
