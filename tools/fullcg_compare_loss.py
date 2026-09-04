#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare two training runs (native warmup+capture vs. persisted LOAD) for numerical parity.

Two input kinds:

* JSONL dumps written by ``MCORE_FULLCG_LOSS_DUMP=<prefix>`` (exact float bits, bitwise check):

      python tools/fullcg_compare_loss.py dump native.rank0.jsonl load.rank0.jsonl \\
          --skip-a 3     # skip the native run's warmup iterations (they are eager, not graph replays)

* Megatron stdout logs (6-decimal ``lm loss`` / 3-decimal ``grad norm``, approximate check):

      python tools/fullcg_compare_loss.py log native.log load.log --skip-a 3

Iteration alignment: dumps record the *global* training iteration (``args.curr_iteration``), so
runs that resumed from different checkpoints line up automatically; only iterations present in
both files are compared. ``--skip-a N`` reports but does not require bitwise equality for
iterations below N (use it to exclude the reference run's eager warmup iterations, whose numerics
may legitimately differ from graph replays of the same math).
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys

LOSS_RE = re.compile(r"iteration\s+(\d+)/\s*\d+.*?lm loss: ([0-9.E+-]+)(?:.*?grad norm: ([0-9.]+))?")


def read_dump(path: str, stage: str = "training") -> dict[int, list]:
    out: dict[int, list] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("stage") != stage:
                continue
            out[int(rec["iteration"])] = rec["values"]
    return out


def bits_to_float(h) -> float:
    return struct.unpack("<d", bytes.fromhex(h))[0] if isinstance(h, str) else float(h)


def compare_dumps(a: dict[int, list], b: dict[int, list], skip_a: int) -> dict:
    common = sorted(set(a) & set(b))
    report = {"iterations_compared": 0, "bitwise_equal": 0, "mismatches": [], "max_abs_diff": 0.0}
    for it in common:
        va, vb = a[it], b[it]
        if len(va) != len(vb):
            report["mismatches"].append({"iteration": it, "reason": f"{len(va)} vs {len(vb)} tensors"})
            continue
        report["iterations_compared"] += 1
        equal = True
        for ta, tb in zip(va, vb):
            if ta["bits"] != tb["bits"]:
                equal = False
                for x, y in zip(ta["bits"], tb["bits"]):
                    if x != y and isinstance(x, str):
                        report["max_abs_diff"] = max(report["max_abs_diff"], abs(bits_to_float(x) - bits_to_float(y)))
        if equal:
            report["bitwise_equal"] += 1
        elif it >= skip_a:
            report["mismatches"].append({"iteration": it, "reason": "bits differ"})
    report["strict_pass"] = (
        report["iterations_compared"] > 0
        and not [m for m in report["mismatches"] if m["iteration"] >= skip_a]
    )
    return report


def read_log(path: str) -> dict[int, tuple[float, float | None]]:
    out = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = LOSS_RE.search(line)
            if m:
                out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)) if m.group(3) else None)
    return out


def compare_logs(a, b, skip_a: int) -> dict:
    common = sorted(set(a) & set(b))
    rep = {"iterations_compared": len(common), "loss_equal_6dp": 0, "mismatches": []}
    for it in common:
        la, ga = a[it]
        lb, gb = b[it]
        if la == lb and (ga is None or gb is None or ga == gb):
            rep["loss_equal_6dp"] += 1
        elif it > skip_a:
            rep["mismatches"].append({"iteration": it, "a": (la, ga), "b": (lb, gb)})
    rep["pass"] = rep["iterations_compared"] > 0 and not rep["mismatches"]
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--skip-a", type=int, default=0, help="iterations below this index are reported but not required to match")
    p.add_argument("--stage", default="training")
    p = sub.add_parser("log")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--skip-a", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "dump":
        rep = compare_dumps(read_dump(args.a, args.stage), read_dump(args.b, args.stage), args.skip_a)
        print(json.dumps(rep, indent=2))
        return 0 if rep["strict_pass"] else 1
    rep = compare_logs(read_log(args.a), read_log(args.b), args.skip_a)
    print(json.dumps(rep, indent=2))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
