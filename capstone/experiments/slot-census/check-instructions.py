#!/usr/bin/env python3
"""Check slotins.c's instruction counts against insctl.c's known loop.

usage: check-instructions.py CONTROL_DIR
insctl's loop is 2 instructions per iteration, so a run with 10^8 iterations must count 2x10^8 more than a run with
none: one thread; four threads (8x10^8, two of them still alive at exit); and a fork, where the parent counts 4x10^8
(its thread's and its own) and the child 4x10^8 in its own, forked line, none of the thread's. Each check is a
difference between two runs (CONTROL_DIR/ins-<mode>-<iterations>.jsonl), so a client that counts nothing, or counts
something fixed, fails it. Tolerance: 1,000 instructions. Exit 0 only if every check holds."""
import json
import os
import sys

ITERATIONS = 100000000
TOLERANCE = 1000
CHECKS = (("single", 0, 2 * ITERATIONS), ("threads", 0, 8 * ITERATIONS), ("fork", 1, 4 * ITERATIONS),
          ("fork", 0, 4 * ITERATIONS))   # mode, forked line, instructions the loop must add


def count(directory, mode, iterations, forked):
    """the instructions of the one line with this fork flag, or None"""
    path = os.path.join(directory, f"ins-{mode}-{iterations}.jsonl")
    lines = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    found = [r["instructions"] for r in lines if r["forked"] == forked]
    return found[0] if len(found) == 1 else None


def main():
    directory = sys.argv[1]
    ok = True
    for mode, forked, want in CHECKS:
        a, b = count(directory, mode, 0, forked), count(directory, mode, ITERATIONS, forked)
        good = a is not None and b is not None and abs((b - a) - want) <= TOLERANCE
        delta = None if a is None or b is None else b - a
        print(f"{'PASS' if good else 'FAIL'} {mode} forked={forked}: {a} -> {b}, delta {delta}, want {want}")
        ok &= good
    print("CONTROL PASS" if ok else "CONTROL FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
