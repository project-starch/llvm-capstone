#!/usr/bin/env python3
"""matrix-compare.py <reference out> <board out>: the mruby/perl release-differential matrix.tsv of a board arm against
its QEMU reference, cell by cell on (outcome, cause, function, in_fault_sites), the fields the compiler lane's own
reference compare uses (the arm column is left out: a board bundle records vp2-<arm>). Prints each differing case and
a last line "<n> of <m> cells the same". Exit 0 when every case is present on both sides and equal, 1 when any differs
or is missing, 2 when either matrix is missing or empty (no data is an error)."""
import csv
import sys
from pathlib import Path

F = ("outcome", "cause", "function", "in_fault_sites")


def load(d):
    p = Path(d) / "matrix.tsv"
    if not p.is_file():
        return None
    return {r["case"]: tuple(r.get(k, "") for k in F) for r in csv.DictReader(p.open(), delimiter="\t")}


a, b = load(sys.argv[1]), load(sys.argv[2])
if not a or not b:
    print(f"NO DATA: reference {len(a or {})} rows, board {len(b or {})} rows"); sys.exit(2)
keys = sorted(set(a) | set(b))
diff = [k for k in keys if a.get(k) != b.get(k)]
for k in diff:
    print(f"  {k}: reference {a.get(k)} board {b.get(k)}")
print(f"{len(keys) - len(diff)} of {len(keys)} cells the same")
sys.exit(1 if diff else 0)
