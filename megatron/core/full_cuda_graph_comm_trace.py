# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Record the torch.distributed collectives issued while a full-iteration graph is captured.

Why: NCCL collectives captured inside a CUDA graph cannot be rebuilt from graph metadata alone
(the kernel nodes reference NCCL's per-communicator plans and buffers, and NCCL attaches host
callbacks / user objects). The Phase-2 plan replaces those nodes with placeholders at SAVE and
re-captures each collective at LOAD, which requires knowing the *semantic* call sequence:
op, communicator, buffers, sizes, peers. Graph JSON has none of that, so we record it here.

This module only records. It is enabled in SAVE mode (``MCORE_FULLCG_COMM_TRACE=0`` disables)
and writes ``comm_trace_<stage>.json`` next to the graph. ``tools/fullcg_graph_inspect.py comm``
compares the recorded launch count with the number of NCCL kernel nodes in the graph.

Coverage note: only calls that go through ``torch.distributed.<fn>`` attribute lookup at call
time are seen. Code that bound the function at import time (``from torch.distributed import
all_reduce``) or calls ``ProcessGroup`` methods directly is missed; the count comparison in the
inspector exposes such gaps.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

TRACE_VERSION = 1

WRAPPED_FUNCTIONS = (
    "all_reduce",
    "reduce",
    "broadcast",
    "all_gather",
    "all_gather_into_tensor",
    "_all_gather_base",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "_reduce_scatter_base",
    "all_to_all",
    "all_to_all_single",
    "gather",
    "scatter",
    "send",
    "recv",
    "isend",
    "irecv",
    "batch_isend_irecv",
    "barrier",
)


def _tensor_desc(t: torch.Tensor) -> dict:
    return {
        "ptr": int(t.data_ptr()),
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": int(t.numel()),
        "device": str(t.device),
    }


def _describe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor": _tensor_desc(value)}
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, torch.Tensor) for v in value):
        return {"tensors": [_tensor_desc(v) for v in value]}
    if isinstance(value, (list, tuple)) and all(isinstance(v, int) for v in value):
        return list(value)
    if isinstance(value, dist.ProcessGroup):
        return {"group": _group_desc(value)}
    if isinstance(value, dist.ReduceOp):
        return {"reduce_op": str(value)}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return {"repr": repr(value)[:80]}


def _group_desc(group) -> dict:
    try:
        if group is None:
            ranks = list(range(dist.get_world_size()))
        else:
            ranks = list(dist.get_process_group_ranks(group))
    except Exception:  # noqa: BLE001
        ranks = []
    name = None
    try:
        name = group.group_name if group is not None else "default"
    except Exception:  # noqa: BLE001
        pass
    return {"ranks": ranks, "name": name}


class CommTraceRecorder:
    """Context manager that logs torch.distributed calls issued inside its scope."""

    def __init__(self):
        self.entries: list[dict] = []
        self._orig: dict[str, Any] = {}

    # ------------------------------------------------------------------ recording
    def _record(self, op: str, args, kwargs):
        entry: dict = {"seq": len(self.entries), "op": op, "args": [], "kwargs": {}}
        group = kwargs.get("group")
        for a in args:
            entry["args"].append(_describe(a))
            if isinstance(a, dist.ProcessGroup):
                group = a
        for k, v in kwargs.items():
            entry["kwargs"][k] = _describe(v)
        if op == "batch_isend_irecv" and args:
            p2p = []
            for o in args[0]:
                try:
                    p2p.append(
                        {
                            "op": getattr(o.op, "__name__", str(o.op)),
                            "peer": int(o.peer) if o.peer is not None else None,
                            "tensor": _tensor_desc(o.tensor),
                            "group": _group_desc(o.group),
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    p2p.append({"repr": repr(o)[:80], "error": str(e)})
            entry["p2p_ops"] = p2p
            groups = {tuple(p.get("group", {}).get("ranks", [])) for p in p2p if "group" in p}
            if len(groups) == 1:
                entry["group_ranks"] = list(next(iter(groups)))
        if "group_ranks" not in entry:
            entry["group_ranks"] = _group_desc(group)["ranks"]
        entry["async_op"] = bool(kwargs.get("async_op", False))
        self.entries.append(entry)

    def _wrap(self, name: str, orig):
        recorder = self

        @functools.wraps(orig)
        def wrapper(*args, **kwargs):
            try:
                recorder._record(name, args, kwargs)
            except Exception as e:  # noqa: BLE001 - never let tracing break the capture
                logger.debug("[fullcg-comm-trace] failed to record %s: %s", name, e)
            return orig(*args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self):
        for name in WRAPPED_FUNCTIONS:
            orig = getattr(dist, name, None)
            if orig is None or name in self._orig:
                continue
            self._orig[name] = orig
            setattr(dist, name, self._wrap(name, orig))
        return self

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(dist, name, orig)
        self._orig.clear()
        return False

    # ------------------------------------------------------------------ output
    def summary(self) -> dict:
        by_op: dict[str, int] = {}
        for e in self.entries:
            by_op[e["op"]] = by_op.get(e["op"], 0) + 1
        return {"num_entries": len(self.entries), "by_op": by_op}

    def to_json(self) -> dict:
        return {
            "version": TRACE_VERSION,
            "rank": dist.get_rank() if dist.is_initialized() else 0,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "entries": self.entries,
        }

    def save(self, path: str) -> str:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_json(), f, indent=1)
        os.replace(tmp, path)
        return path
