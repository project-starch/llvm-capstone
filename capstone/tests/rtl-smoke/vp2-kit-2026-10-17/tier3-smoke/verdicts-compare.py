#!/usr/bin/env python3
"""verdicts-compare.py <reference out> <board out>: a cpython (or any verdicts.jsonl) board arm against its QEMU reference,
cell by cell on (verdict, attribution, fault cause, fault symbol, fault offset), the fields the compiler lane's own batch
reference compare uses (batch-refs/run-pass-r60.sh cmp()). Each <out> is a directory holding verdicts.jsonl. Prints each
differing case and a last line "<n> of <m> cells the same". Exit 0 when every case is present on both sides and equal,
1 when any differs or is missing, 2 when either side has no verdicts (no data is an error)."""
import json
import sys
from pathlib import Path


def load(d):
    p = Path(d) / "verdicts.jsonl"
    if not p.is_file():
        return None
    rows = {}
    for line in p.open():
        if line.strip():
            r = json.loads(line)
            o = r.get("observation", {})
            f = o.get("fault") or {}
            rows[o.get("case")] = (r.get("verdict"), o.get("attribution"), f.get("cause"), f.get("symbol"), f.get("offset"))
    return rows


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    if not a or not b:
        print(f"NO DATA: reference {0 if not a else len(a)} cells, board {0 if not b else len(b)} cells"); sys.exit(2)
    keys = sorted(set(a) | set(b))
    diff = [k for k in keys if a.get(k) != b.get(k)]
    for k in diff:
        print(f"DIFF {k}: reference {a.get(k)} board {b.get(k)}")
    print(f"{len(keys) - len(diff)} of {len(keys)} cells the same")
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
