# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for tools/fullcg_graph_inspect.py using synthetic graph JSON (no GPU / torch)."""

import importlib.util
import json
import os
import struct
import tempfile

_TOOL = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "fullcg_graph_inspect.py")
_spec = importlib.util.spec_from_file_location("fullcg_graph_inspect", _TOOL)
insp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(insp)


def _kernel(nid, name, params_hex=(), binary_hash=1, attrs=None):
    return {
        "id": nid,
        "type": "KernelNode",
        "params": {
            "function_name": name,
            "kernel_source_binary_hash": binary_hash,
            "kernelParams": [
                {"index": i, "offset": 8 * i, "size": 8, "value_hex": h} for i, h in enumerate(params_hex)
            ],
            "extra_argBuffer_hex": "",
            "kernel_node_attrs": attrs or {},
        },
    }


def _graph(ptr_a, ptr_b):
    return {
        "nodes": [
            _kernel(0, "_ZN8cutlass3GemmE", [struct.pack("<Q", ptr_a).hex()], binary_hash=11),
            _kernel(1, "ncclDevKernel_AllReduce_Sum_bf16_RING_LL", [struct.pack("<Q", ptr_b).hex()], binary_hash=22),
            _kernel(2, "flash_fwd_kernel", [], binary_hash=33,
                    attrs={"programmaticStreamSerializationAllowed": 1}),
            _kernel(3, "my_mystery_kernel", [], binary_hash=44),
            {"id": 4, "type": "MemcpyNode",
             "params": {"srcMemoryType": 1, "dstMemoryType": 2, "srcDevice": 0, "dstDevice": ptr_a}},
            {"id": 5, "type": "MemsetNode", "params": {"dst": ptr_b}},
            {"id": 6, "type": "EmptyNode", "params": {}},
        ],
        "dependencies": [
            {"from": 0, "to": 1},
            {"from": 1, "to": 2, "type": 1, "from_port": 0, "to_port": 1},
        ],
        "generators": [{"id": 0, "name": "default", "seed": 1, "wholegraph_increment": 4,
                        "seed_extragraph_ptr": ptr_a, "offset_extragraph_ptr": ptr_a + 8}],
        "allocator_events": {"start_base_addr": 0x500000000000,
                             "events": [{"type": "alloc", "size": 4096, "ptr": 1}, {"type": "free", "ptr": 1}]},
        "output_tensors": {"type": 2, "tensors": [{"data_ptr": ptr_a}]},
    }


def test_classify_kernel():
    assert insp.classify_kernel("ncclDevKernel_Generic") == "nccl"
    assert insp.classify_kernel("_ZN8cutlass6KernelINS_4gemm") == "cublas/gemm"
    assert insp.classify_kernel("nvshmemi_put_kernel") == "nvshmem"
    assert insp.classify_kernel("deep_ep::intranode::dispatch") == "deepep/hybridep"
    assert insp.classify_kernel("_ZN2at6native29vectorized_elementwise_kernel") == "torch"
    assert insp.classify_kernel("totally_unknown") == "unknown"


def test_summarize_graph_and_verdicts():
    g = _graph(0x500000001000, 0x500000002000)
    s = insp.summarize_graph(g, demangle=False)
    assert s["num_nodes"] == 7
    assert s["node_types"]["KernelNode"] == 4
    assert s["kernel_libraries"]["nccl"] == 1
    assert s["nccl_kernel_nodes"] == [1]
    assert s["host_memcpy_nodes"] == [4]
    assert s["memcpy_kinds"] == {"host->device": 1}
    assert s["num_pdl_edges"] == 1 and s["num_pdl_kernel_attrs"] == 1
    assert s["rng_consumed"] is True
    assert s["generators"][0]["has_extragraph_ptrs"] is True
    assert s["allocator_events"]["alloc_bytes"] == 4096 and s["allocator_events"]["num_frees"] == 1
    assert s["num_binaries"] == 4
    assert ("my_mystery_kernel", 1) in s["unknown_kernels"]
    v = "\n".join(insp.verdicts(s, {"pointer_report": {"num_pointers": {"unmanaged": 2, "warmup": 1},
                                                      "unmanaged_pointers": ["0x1", "0x2"]},
                                    "required_intervals": [[1, 2]], "warnings": ["w1"]}))
    assert "Phase 2" in v and "host memcpy" in v and "RNG: consumed" in v
    assert "2 graph pointers outside" in v and "SAVE warning: w1" in v


def test_verdicts_clean_graph():
    g = _graph(0x500000001000, 0x500000002000)
    g["nodes"] = [n for n in g["nodes"] if n["id"] not in (1, 4)]
    g["dependencies"] = [{"from": 0, "to": 2}]
    s = insp.summarize_graph(g, demangle=False)
    v = "\n".join(insp.verdicts(s, None))
    assert "Phase 1 restore applicable" in v and "host memcpy" not in v


def test_diff_graphs_detects_argument_nondeterminism():
    a = _graph(0x500000001000, 0x500000002000)
    b = _graph(0x500000001000, 0x500000002000)
    assert insp.diff_graphs(a, b)["deterministic"] is True
    c = _graph(0x500000001000, 0x500000009000)  # pointer arg of node 1 differs, memset dst differs
    d = insp.diff_graphs(a, c)
    assert d["structure_equal"] is True
    assert d["nodes_with_arg_diffs"] == 2
    assert d["deterministic"] is False
    assert any(x["node"] == 1 for x in d["arg_diffs"])
    e = _graph(0x500000001000, 0x500000002000)
    e["nodes"].append(_kernel(7, "extra"))
    assert insp.diff_graphs(a, e)["structure_equal"] is False


def test_compare_comm_trace_counts():
    s = insp.summarize_graph(_graph(1 << 44, 2 << 44), demangle=False)
    trace = {"entries": [
        {"op": "all_reduce", "group_ranks": [0, 1]},
        {"op": "batch_isend_irecv", "group_ranks": [0, 1], "p2p_ops": [{}, {}]},
    ]}
    r = insp.compare_comm_trace(trace, s)
    assert r["estimated_nccl_launches"] == 2 and r["nccl_kernel_nodes_in_graph"] == 1
    assert r["aligned"] is False
    trace["entries"].pop()
    assert insp.compare_comm_trace(trace, s)["aligned"] is True


def test_manifest_diff_and_cli_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        ma = {"fingerprint": "x", "stages": {"training": {
            "phase_offsets": {"training_first_call": 10}, "required_intervals": [[1, 2]],
            "pointer_report": {"num_pointers": {"warmup": 1}}, "num_tensors": 3}}}
        mb = json.loads(json.dumps(ma))
        mb["stages"]["training"]["phase_offsets"]["training_first_call"] = 11
        for name, m in (("a", ma), ("b", mb)):
            os.makedirs(os.path.join(d, name))
            with open(os.path.join(d, name, "fullcg_manifest.json"), "w") as f:
                json.dump(m, f)
        res = insp.diff_manifests(ma, mb)
        assert res["stages"]["training"]["phase_offsets_equal"] is False
        assert insp.main(["manifest-diff", os.path.join(d, "a"), os.path.join(d, "b")]) == 1
        # inspect via CLI on a graph file
        gpath = os.path.join(d, "a", "graph_training.json")
        with open(gpath, "w") as f:
            json.dump(_graph(0x500000001000, 0x500000002000), f)
        assert insp.main(["inspect", gpath, "--json", "--no-demangle"]) == 0
