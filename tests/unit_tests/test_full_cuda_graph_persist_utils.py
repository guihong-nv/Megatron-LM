# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-Python tests for megatron.core.full_cuda_graph_persist_utils (no GPU / torch needed)."""

import json
import os
import struct
import tempfile

import pytest

from megatron.core import full_cuda_graph_persist_utils as U


class FakeTensor:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"FakeTensor({self.name})"


def _is_tensor(x):
    return isinstance(x, FakeTensor)


def test_flatten_unflatten_roundtrip_nested():
    a, b, c = FakeTensor("a"), FakeTensor("b"), FakeTensor("c")
    obj = [
        {"lm loss": a, "num_tokens": 17, "flag": True, "ratio": 0.5, "name": "x", "nothing": None},
        {"lm loss": b, "aux": (c, 3)},
    ]
    schema, tensors = U.flatten_struct(obj, _is_tensor)
    assert tensors == [a, b, c]
    assert U.count_tensors(schema) == 3
    # schema must be JSON serializable
    json.dumps(schema)
    rebuilt = U.unflatten_struct(schema, tensors)
    assert rebuilt[0]["lm loss"] is a
    assert rebuilt[0]["num_tokens"] == 17 and isinstance(rebuilt[0]["num_tokens"], int)
    assert rebuilt[0]["flag"] is True
    assert rebuilt[0]["ratio"] == 0.5
    assert rebuilt[0]["nothing"] is None
    assert rebuilt[1]["aux"] == (c, 3) and isinstance(rebuilt[1]["aux"], tuple)


def test_flatten_rejects_unknown_objects():
    with pytest.raises(TypeError):
        U.flatten_struct([object()], _is_tensor)


def test_flatten_rejects_non_str_keys():
    with pytest.raises(TypeError):
        U.flatten_struct({1: FakeTensor("a")}, _is_tensor)


def test_fingerprint_is_stable_and_filters_volatile_keys():
    args = {
        "seq_length": 4096,
        "micro_batch_size": 1,
        "rank": 3,
        "load": "/ckpt/x",
        "save_interval": 100,
        "tensorboard_dir": "/tb",
        "seed": 1234,
        "num_layers": 61,
        "some_list": [1, "a", None],
        "some_obj": object(),
        "_private": 1,
    }
    args["moe_router_load_balancing_type"] = "aux_loss"
    args["lr"] = 1e-4
    args["train_iters"] = 100
    fields = U.select_fingerprint_fields(args)
    assert "rank" not in fields and "load" not in fields and "seed" not in fields
    assert "seq_length" in fields and "num_layers" in fields
    assert "_private" not in fields
    # whole-key regexes: model-structure keys containing "load" survive, schedule keys do not
    assert "moe_router_load_balancing_type" in fields
    assert "lr" not in fields and "train_iters" not in fields and "save_interval" not in fields
    assert U.is_fingerprint_denied("tensorboard_dir") and not U.is_fingerprint_denied("hidden_size")
    fp1 = U.compute_fingerprint(fields)
    args2 = dict(args, rank=7, load="/other", seed=99)
    fp2 = U.compute_fingerprint(U.select_fingerprint_fields(args2))
    assert fp1 == fp2
    args3 = dict(args, seq_length=8192)
    fp3 = U.compute_fingerprint(U.select_fingerprint_fields(args3))
    assert fp3 != fp1
    diffs = U.diff_fingerprint_fields(fields, U.select_fingerprint_fields(args3))
    assert any(d.startswith("seq_length") for d in diffs)


def _hex_words(*vals):
    return b"".join(struct.pack("<Q", v) for v in vals).hex()


def test_extract_candidate_pointers_kernel_and_memops():
    lo, hi = 0x500000000000, 0x500000000000 + (1 << 40)
    p_in = lo + 0x1000
    p_in2 = lo + 0x200000
    p_out = 0x7F0000001000
    graph = {
        "nodes": [
            {
                "id": 0,
                "type": "KernelNode",
                "params": {
                    "kernelParams": [
                        {"index": 0, "value_hex": _hex_words(p_in)},
                        {"index": 1, "value_hex": _hex_words(42)},
                    ],
                    "extra_argBuffer_hex": "",
                },
            },
            {
                "id": 1,
                "type": "KernelNode",
                "params": {
                    "kernelParams": [],
                    # pointer packed at a 4-byte (not 8-byte) aligned offset inside an opaque struct
                    "extra_argBuffer_hex": (struct.pack("<I", 7) + struct.pack("<Q", p_in2)).hex(),
                },
            },
            {"id": 2, "type": "MemcpyNode", "params": {"srcDevice": p_out, "dstDevice": p_in}},
            {"id": 3, "type": "MemsetNode", "params": {"dst": p_in2}},
            {"id": 4, "type": "EmptyNode", "params": {}},
        ]
    }
    found = U.extract_candidate_pointers(graph, lo, hi)
    assert set(found) == {p_in, p_in2}
    assert any(w.startswith("node0:param0") for w in found[p_in])
    assert any(w.startswith("node2:dstDevice") for w in found[p_in])
    assert any(w.startswith("node1:argbuf@4") for w in found[p_in2])
    assert any(w.startswith("node3:dst") for w in found[p_in2])


def test_segments_and_classification():
    base = 0x500000000000
    MB = 1 << 20
    snapshot = [
        {"address": base - 100 * MB, "total_size": 2 * MB, "segment_pool_id": (0, 0), "device": 1},  # other GPU
        {"address": base - 50 * MB, "total_size": 2 * MB, "segment_pool_id": (0, 0)},  # scratch
        {"address": base + 0 * MB, "total_size": 20 * MB, "segment_pool_id": (0, 0)},  # init
        {"address": base + 40 * MB, "total_size": 2 * MB, "segment_pool_id": [0, 0]},  # warmup
        {"address": base + 100 * MB, "total_size": 20 * MB, "segment_pool_id": (3, 0)},  # private
        {"address": base + 200 * MB, "total_size": 4 * MB, "segment_pool_id": (0, 0)},  # capture
    ]
    segs = U.segments_from_snapshot(snapshot, device=0)
    assert [s[2] for s in segs] == ["default", "default", "default", "private", "default"]
    init_end = base + 30 * MB
    capture_start = base + 150 * MB
    pointers = [
        base + 1 * MB,  # init
        base + 41 * MB,  # warmup
        base + 41 * MB + 4096,  # warmup, same segment
        base + 101 * MB,  # private pool -> capture
        base + 201 * MB,  # after capture start -> capture
        base + 60 * MB,  # no segment -> unmanaged
        base - 49 * MB,  # scratch prefix -> non-reproducible
    ]
    cls = U.classify_pointers(pointers, segs, init_end, capture_start, scratch_end_addr=base)
    assert len(cls["scratch"]) == 1
    assert len(cls["init"]) == 1
    assert len(cls["warmup"]) == 2
    assert len(cls["capture"]) == 2
    assert len(cls["unmanaged"]) == 1
    req = U.required_intervals(cls)
    assert req == [(base + 40 * MB, 2 * MB)]
    report = U.summarize_pointer_report(cls)
    assert report["num_pointers"]["warmup"] == 2 and report["num_pointers"]["scratch"] == 1
    assert report["scratch_segments"] == [[base - 50 * MB, 2 * MB]]
    assert report["warmup_segments"] == [[base + 40 * MB, 2 * MB]]
    assert report["unmanaged_pointers"] == [hex(base + 60 * MB)]


def test_merge_intervals():
    assert U.merge_intervals([(10, 5), (15, 5), (30, 1), (0, 3)]) == [(0, 3), (10, 10), (30, 1)]
    assert U.merge_intervals([(10, 5), (12, 1)]) == [(10, 5)]
    assert U.merge_intervals([]) == []


def test_align_up():
    assert U.align_up(1) == U.ALLOC_ALIGNMENT
    assert U.align_up(U.ALLOC_ALIGNMENT) == U.ALLOC_ALIGNMENT
    assert U.align_up(U.ALLOC_ALIGNMENT + 1) == 2 * U.ALLOC_ALIGNMENT


def test_manifest_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        m = U.build_manifest("abc", {"seq_length": 1}, 0x500000000000, 1 << 40, "rank0", {"torch": "2.11"})
        U.add_stage_to_manifest(
            m,
            "training",
            result_schema={"t": "none"},
            static_buffers_schema={"t": "none"},
            num_tensors=0,
            generator_names=["default", "model-parallel-rng"],
            phase_offsets={"init_end": 123, "capture_start": 456},
            required=[(1, 2)],
            pointer_report={},
            warnings=["w"],
        )
        path = U.write_manifest(d, m)
        assert os.path.basename(path) == U.MANIFEST_FILENAME
        back = U.read_manifest(d)
        assert back["fingerprint"] == "abc"
        assert back["stages"]["training"]["required_intervals"] == [[1, 2]]
        assert back["stages"]["training"]["graph_json"] == "fullcg_training.json"
        assert U.read_manifest(os.path.join(d, "missing")) is None
