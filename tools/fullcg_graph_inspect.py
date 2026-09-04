#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Inspect / diff Foundry graph archives produced by ``--cuda-graph-persist-mode save``.

Pure Python (no torch, no GPU) so it can run anywhere the archive is readable.

    # what is inside the captured graph? (node types, which libraries the kernels come from,
    # NCCL/host-memcpy/RNG flags, PDL edges, pointer provenance from the manifest)
    python tools/fullcg_graph_inspect.py inspect /nvme/fullcg_archive/rank0/fullcg_training.json

    # are two SAVE runs of the same config deterministic? (structure + kernel arguments)
    python tools/fullcg_graph_inspect.py diff runA/rank0/fullcg_training.json runB/rank0/fullcg_training.json

    # did the allocation trajectory / required intervals change between two SAVE runs?
    python tools/fullcg_graph_inspect.py manifest-diff runA/rank0 runB/rank0

    # Phase-2 go/no-go: how many NCCL kernels are in the graph vs. how many collectives were
    # recorded during capture (comm_trace_<stage>.json written by the SAVE run)?
    python tools/fullcg_graph_inspect.py comm /nvme/fullcg_archive/rank0 --stage training
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Iterable, Optional

# Library classification by (possibly mangled) kernel name. Order matters: first match wins.
KERNEL_LIBRARY_PATTERNS: list[tuple[str, str]] = [
    ("nccl", r"nccl"),
    ("nvshmem", r"nvshmem"),
    ("deepep/hybridep", r"deep_ep|deepep|hybrid_ep|hybridep|internode|intranode"),
    ("cudnn/attention", r"cudnn|flash_fwd|flash_bwd|fmha|fused_attn|sdpa"),
    ("cublas/gemm", r"cublas|nvjet|xmma|cutlass|gemm|Gemm|GEMM"),
    ("transformer_engine", r"transformer_engine|nvte|te_kernels|Nvte"),
    ("triton", r"triton"),
    ("torch", r"at_cuda_detail|at::native|at6native|c10|vectorized_elementwise|elementwise_kernel|reduce_kernel|"
              r"index_elementwise|cub::|cub6|CatArrayBatchedCopy|unrolled_elementwise|distribution_elementwise|"
              r"softmax_warp|cunn_|indexSelect|gather_kernel|scatter_gather"),
    ("apex/megatron_fused", r"fused_|megatron|Megatron|apex"),
]

CU_MEMORYTYPE = {0: "unspecified", 1: "host", 2: "device", 3: "array", 4: "unified"}


def _demangle(names: Iterable[str]) -> dict[str, str]:
    """Best-effort batch demangle via c++filt (falls back to identity)."""
    names = list(dict.fromkeys(names))
    if not names or shutil.which("c++filt") is None:
        return {n: n for n in names}
    try:
        out = subprocess.run(
            ["c++filt"], input="\n".join(names) + "\n", capture_output=True, text=True, timeout=30
        ).stdout.splitlines()
        if len(out) == len(names):
            return dict(zip(names, out))
    except Exception:  # noqa: BLE001
        pass
    return {n: n for n in names}


def classify_kernel(name: str) -> str:
    for lib, pat in KERNEL_LIBRARY_PATTERNS:
        if re.search(pat, name):
            return lib
    return "unknown"


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------------------ inspect


def summarize_graph(gj: dict, demangle: bool = True) -> dict:
    nodes = gj.get("nodes", [])
    type_hist = collections.Counter(n.get("type", "?") for n in nodes)
    kernel_nodes = [n for n in nodes if n.get("type") == "KernelNode"]
    raw_names = [n["params"].get("function_name", "") for n in kernel_nodes]
    dm = _demangle(raw_names) if demangle else {n: n for n in raw_names}

    lib_hist: collections.Counter = collections.Counter()
    per_kernel: collections.Counter = collections.Counter()
    per_binary: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    nccl_nodes, unknown_names = [], collections.Counter()
    pdl_kernels = 0
    for n, raw in zip(kernel_nodes, raw_names):
        name = dm.get(raw, raw)
        lib = classify_kernel(name) if classify_kernel(name) != "unknown" else classify_kernel(raw)
        lib_hist[lib] += 1
        per_kernel[name] += 1
        per_binary[int(n["params"].get("kernel_source_binary_hash", 0))][name] += 1
        if lib == "nccl":
            nccl_nodes.append(n["id"])
        if lib == "unknown":
            unknown_names[name] += 1
        attrs = n["params"].get("kernel_node_attrs", {}) or {}
        if attrs.get("programmaticStreamSerializationAllowed") or attrs.get("programmaticEvent"):
            pdl_kernels += 1

    memcpy_kinds: collections.Counter = collections.Counter()
    host_memcpy_nodes = []
    for n in nodes:
        if n.get("type") == "MemcpyNode":
            p = n["params"]
            kind = f"{CU_MEMORYTYPE.get(int(p.get('srcMemoryType', 0)), '?')}->" \
                   f"{CU_MEMORYTYPE.get(int(p.get('dstMemoryType', 0)), '?')}"
            memcpy_kinds[kind] += 1
            if 1 in (int(p.get("srcMemoryType", 0)), int(p.get("dstMemoryType", 0))):
                host_memcpy_nodes.append(n["id"])

    deps = gj.get("dependencies", [])
    pdl_edges = sum(1 for d in deps if int(d.get("type", 0)) != 0 or int(d.get("to_port", 0)) != 0)

    gens = gj.get("generators", [])
    ev = gj.get("allocator_events", {}) or {}
    events = ev.get("events", [])
    alloc_bytes = sum(int(e.get("size", 0)) for e in events if e.get("type") == "alloc")

    out_t = (gj.get("output_tensors", {}) or {}).get("tensors", [])

    return {
        "num_nodes": len(nodes),
        "node_types": dict(type_hist),
        "num_kernel_nodes": len(kernel_nodes),
        "num_unique_kernels": len(per_kernel),
        "kernel_libraries": dict(lib_hist.most_common()),
        "top_kernels": per_kernel.most_common(25),
        "unknown_kernels": unknown_names.most_common(25),
        "num_binaries": len(per_binary),
        "nccl_kernel_nodes": nccl_nodes,
        "host_memcpy_nodes": host_memcpy_nodes,
        "memcpy_kinds": dict(memcpy_kinds),
        "num_dependencies": len(deps),
        "num_pdl_edges": pdl_edges,
        "num_pdl_kernel_attrs": pdl_kernels,
        "generators": [
            {
                "name": g.get("name", ""),
                "id": g.get("id"),
                "wholegraph_increment": g.get("wholegraph_increment", 0),
                "has_extragraph_ptrs": bool(g.get("seed_extragraph_ptr")),
            }
            for g in gens
        ],
        "rng_consumed": any(int(g.get("wholegraph_increment", 0)) > 0 for g in gens),
        "allocator_events": {
            "start_base_addr": ev.get("start_base_addr"),
            "num_events": len(events),
            "alloc_bytes": alloc_bytes,
            "num_frees": sum(1 for e in events if e.get("type") == "free"),
            "num_reserves": sum(1 for e in events if e.get("type") == "reserve"),
        },
        "num_output_tensors": len(out_t),
        "topology_key": gj.get("topology_key", ""),
    }


def verdicts(summary: dict, manifest_stage: Optional[dict]) -> list[str]:
    """Human-readable go/no-go notes for the persistence phases."""
    v = []
    if summary["nccl_kernel_nodes"]:
        v.append(
            f"NCCL: {len(summary['nccl_kernel_nodes'])} NCCL kernel nodes inside the graph -> Phase 1 "
            "restore is not applicable (PP/DP/TP comm inside the captured region); needs Phase 2 "
            "hybrid re-capture. Check comm_trace alignment with `comm`."
        )
    else:
        v.append("NCCL: none inside the graph -> Phase 1 restore applicable.")
    if summary["host_memcpy_nodes"]:
        v.append(
            f"host memcpy: {len(summary['host_memcpy_nodes'])} memcpy nodes touch host memory; their "
            "host addresses are process-specific and will not survive LOAD. Find the .cpu()/pinned "
            "copies inside the captured region."
        )
    unknown = summary["node_types"].keys() - {
        "KernelNode", "MemcpyNode", "MemsetNode", "EventRecordNode", "EventWaitNode", "EmptyNode"
    }
    if unknown:
        v.append(f"node types not supported by the Foundry serializer: {sorted(unknown)}")
    if summary["rng_consumed"]:
        missing = [g["name"] or f"id{g['id']}" for g in summary["generators"]
                   if g["wholegraph_increment"] and not g["has_extragraph_ptrs"]]
        if missing:
            v.append(f"RNG: consumed by the graph but no extragraph addresses saved for {missing} "
                     "(Foundry without the B1 patch) -> LOAD cannot continue the RNG stream.")
        else:
            v.append("RNG: consumed by the graph; extragraph addresses present -> bind at LOAD.")
    if summary["num_pdl_kernel_attrs"] and not summary["num_pdl_edges"]:
        v.append("PDL: kernels carry programmatic-launch attributes but no edge data was saved "
                 "(Foundry without the A7 patch); restored graph loses PDL overlap.")
    if manifest_stage:
        rep = manifest_stage.get("pointer_report", {}) or {}
        nums = rep.get("num_pointers", {})
        if nums.get("unmanaged"):
            v.append(f"pointers: {nums['unmanaged']} graph pointers outside every caching-allocator "
                     f"segment (e.g. {', '.join((rep.get('unmanaged_pointers') or [])[:4])}); must be "
                     "deterministic by other means (NVSHMEM heap, NCCL buffers, foreign allocs).")
        if nums.get("warmup"):
            v.append(f"pointers: {nums['warmup']} pointers into warmup-phase segments "
                     f"({len(manifest_stage.get('required_intervals', []))} intervals mapped at LOAD).")
        for w in manifest_stage.get("warnings", []):
            v.append("SAVE warning: " + w)
    return v


def cmd_inspect(args) -> int:
    gj = load_json(args.graph_json)
    summary = summarize_graph(gj, demangle=not args.no_demangle)
    stage = None
    mdir = args.manifest or os.path.dirname(os.path.abspath(args.graph_json))
    mpath = os.path.join(mdir, "fullcg_manifest.json")
    if os.path.exists(mpath):
        m = load_json(mpath)
        for st in m.get("stages", {}).values():
            if st.get("graph_json") == os.path.basename(args.graph_json):
                stage = st
    summary["verdicts"] = verdicts(summary, stage)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    print(f"nodes: {summary['num_nodes']}  types: {summary['node_types']}")
    print(f"kernel nodes: {summary['num_kernel_nodes']} unique: {summary['num_unique_kernels']} "
          f"binaries: {summary['num_binaries']}")
    print(f"kernel libraries: {summary['kernel_libraries']}")
    print(f"dependencies: {summary['num_dependencies']} (PDL edges: {summary['num_pdl_edges']}, "
          f"kernels with PDL attrs: {summary['num_pdl_kernel_attrs']})")
    print(f"memcpy kinds: {summary['memcpy_kinds']}")
    print(f"generators: {summary['generators']}")
    print(f"allocator events: {summary['allocator_events']}")
    print(f"output tensors: {summary['num_output_tensors']}")
    print("top kernels:")
    for name, cnt in summary["top_kernels"][: args.top]:
        print(f"  {cnt:6d}  {name[:140]}")
    if summary["unknown_kernels"]:
        print("unclassified kernels:")
        for name, cnt in summary["unknown_kernels"][: args.top]:
            print(f"  {cnt:6d}  {name[:140]}")
    print("verdicts:")
    for v in summary["verdicts"]:
        print("  - " + v)
    return 0


# --------------------------------------------------------------------------------- diff


def diff_graphs(a: dict, b: dict) -> dict:
    """Structural + argument diff of two graph JSONs captured from the same config."""
    na, nb = a.get("nodes", []), b.get("nodes", [])
    res: dict = {"num_nodes": (len(na), len(nb)), "structure_equal": True, "arg_diffs": []}
    if len(na) != len(nb):
        res["structure_equal"] = False
    n_common = min(len(na), len(nb))
    type_mismatch = name_mismatch = 0
    arg_diff_nodes: collections.Counter = collections.Counter()
    for x, y in zip(na[:n_common], nb[:n_common]):
        if x.get("type") != y.get("type"):
            type_mismatch += 1
            continue
        if x.get("type") == "KernelNode":
            px, py = x["params"], y["params"]
            if px.get("function_name") != py.get("function_name"):
                name_mismatch += 1
                continue
            differing = []
            for kx, ky in zip(px.get("kernelParams", []), py.get("kernelParams", [])):
                if kx.get("value_hex") != ky.get("value_hex"):
                    differing.append(kx.get("index"))
            if px.get("extra_argBuffer_hex") != py.get("extra_argBuffer_hex"):
                differing.append("argbuf")
            if differing:
                arg_diff_nodes[px.get("function_name", "?")] += 1
                if len(res["arg_diffs"]) < 50:
                    res["arg_diffs"].append({"node": x["id"], "kernel": px.get("function_name", "?")[:120],
                                             "params": differing})
        elif x.get("type") in ("MemcpyNode", "MemsetNode"):
            if x.get("params") != y.get("params"):
                arg_diff_nodes[x["type"]] += 1
    if type_mismatch or name_mismatch:
        res["structure_equal"] = False
    res["type_mismatch"] = type_mismatch
    res["kernel_name_mismatch"] = name_mismatch
    res["nodes_with_arg_diffs"] = sum(arg_diff_nodes.values())
    res["arg_diff_by_kernel"] = arg_diff_nodes.most_common(30)
    da, db = a.get("dependencies", []), b.get("dependencies", [])
    res["dependencies_equal"] = sorted((d["from"], d["to"]) for d in da) == sorted((d["from"], d["to"]) for d in db)
    ga, gb = a.get("generators", []), b.get("generators", [])
    res["generators_equal"] = [(g.get("name"), g.get("wholegraph_increment")) for g in ga] == \
                              [(g.get("name"), g.get("wholegraph_increment")) for g in gb]
    ea, eb = a.get("allocator_events", {}) or {}, b.get("allocator_events", {}) or {}
    res["allocator_start_equal"] = ea.get("start_base_addr") == eb.get("start_base_addr")
    res["allocator_events_equal"] = ea.get("events") == eb.get("events")
    res["deterministic"] = (res["structure_equal"] and res["nodes_with_arg_diffs"] == 0
                            and res["dependencies_equal"] and res["allocator_events_equal"])
    return res


def cmd_diff(args) -> int:
    res = diff_graphs(load_json(args.a), load_json(args.b))
    print(json.dumps(res, indent=2))
    return 0 if res["deterministic"] else 1


def diff_manifests(ma: dict, mb: dict) -> dict:
    out: dict = {"fingerprint_equal": ma.get("fingerprint") == mb.get("fingerprint"), "stages": {}}
    for stage in sorted(set(ma.get("stages", {})) | set(mb.get("stages", {}))):
        sa, sb = ma.get("stages", {}).get(stage), mb.get("stages", {}).get(stage)
        if sa is None or sb is None:
            out["stages"][stage] = "missing on one side"
            continue
        out["stages"][stage] = {
            "phase_offsets_equal": sa.get("phase_offsets") == sb.get("phase_offsets"),
            "phase_offsets": (sa.get("phase_offsets"), sb.get("phase_offsets")),
            "required_intervals_equal": sa.get("required_intervals") == sb.get("required_intervals"),
            "num_required_intervals": (len(sa.get("required_intervals", [])), len(sb.get("required_intervals", []))),
            "pointer_counts": ((sa.get("pointer_report") or {}).get("num_pointers"),
                               (sb.get("pointer_report") or {}).get("num_pointers")),
            "num_tensors_equal": sa.get("num_tensors") == sb.get("num_tensors"),
        }
    return out


def cmd_manifest_diff(args) -> int:
    ma = load_json(os.path.join(args.a, "fullcg_manifest.json"))
    mb = load_json(os.path.join(args.b, "fullcg_manifest.json"))
    res = diff_manifests(ma, mb)
    print(json.dumps(res, indent=2))
    ok = res["fingerprint_equal"] and all(
        isinstance(s, dict) and s["phase_offsets_equal"] and s["required_intervals_equal"]
        for s in res["stages"].values()
    )
    return 0 if ok else 1


# --------------------------------------------------------------------------------- comm


def compare_comm_trace(trace: dict, summary: dict) -> dict:
    """Phase-2 go/no-go: recorded collectives vs NCCL kernel nodes in the graph."""
    entries = trace.get("entries", [])
    by_op = collections.Counter(e["op"] for e in entries)
    # One NCCL kernel per collective launch; a batch_isend_irecv group launches one kernel per
    # ncclGroupEnd, so count groups rather than individual P2P ops.
    launches = sum(1 for e in entries if e["op"] != "p2p_op")
    return {
        "recorded_entries": len(entries),
        "recorded_by_op": dict(by_op),
        "estimated_nccl_launches": launches,
        "nccl_kernel_nodes_in_graph": len(summary["nccl_kernel_nodes"]),
        "groups": sorted({tuple(e.get("group_ranks", [])) for e in entries}, key=lambda t: (len(t), t))[:20],
        "aligned": launches == len(summary["nccl_kernel_nodes"]),
        "note": "alignment by count only; Phase 2 replaces NCCL nodes with placeholders and re-captures "
                "each recorded collective at LOAD",
    }


def cmd_comm(args) -> int:
    gpath = os.path.join(args.archive_dir, f"fullcg_{args.stage}.json")
    tpath = os.path.join(args.archive_dir, f"comm_trace_{args.stage}.json")
    summary = summarize_graph(load_json(gpath), demangle=not args.no_demangle)
    trace = load_json(tpath) if os.path.exists(tpath) else {"entries": []}
    res = compare_comm_trace(trace, summary)
    print(json.dumps(res, indent=2))
    return 0 if res["aligned"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("graph_json")
    p.add_argument("--manifest", help="archive dir holding fullcg_manifest.json (default: graph dir)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--no-demangle", action="store_true")
    p.set_defaults(func=cmd_inspect)
    p = sub.add_parser("diff")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_diff)
    p = sub.add_parser("manifest-diff")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_manifest_diff)
    p = sub.add_parser("comm")
    p.add_argument("archive_dir")
    p.add_argument("--stage", default="training")
    p.add_argument("--no-demangle", action="store_true")
    p.set_defaults(func=cmd_comm)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
