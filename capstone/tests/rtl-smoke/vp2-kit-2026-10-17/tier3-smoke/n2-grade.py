#!/usr/bin/env python3
"""n2-grade.py <first out> <second out> [--cause-only CASE_PREFIX ...]: a repeat board reading (N=2) graded against the
FIRST board reading of the same arm on the same bitstream, cell by cell on verdicts-compare.py's fields (verdict,
attribution, fault cause, fault symbol, fault offset). A case whose name starts with a --cause-only prefix is a
pre-registered exception (Q-31: silicon raises cause 4 or 13 where the access is the same): it may differ in the CAUSE
field alone, and only between the Q-31 causes {4, 13, 24}; it is reported as ALLOWED, never silently. Any other difference,
and any case present on one side only, is a DIFF. Exit 0 when every difference is ALLOWED, 1 on any DIFF, 2 on no data."""
import json
import sys
from pathlib import Path

Q31 = {4, 13, 24}


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
    argv = sys.argv[1:]
    allow = []
    if "--cause-only" in argv:
        i = argv.index("--cause-only"); allow = argv[i + 1:]; argv = argv[:i]
    if len(argv) != 2:
        print(__doc__); sys.exit(2)
    a, b = load(argv[0]), load(argv[1])
    if not a or not b:
        print(f"NO DATA: first {0 if not a else len(a)} cells, second {0 if not b else len(b)} cells"); sys.exit(2)
    keys = sorted(set(a) | set(b))
    diff = allowed = 0
    for k in keys:
        x, y = a.get(k), b.get(k)
        if x == y:
            continue
        if (x and y and any(k.startswith(p) for p in allow) and x[:2] == y[:2] and x[3:] == y[3:]
                and x[2] in Q31 and y[2] in Q31):
            allowed += 1; print(f"ALLOWED {k}: cause first {x[2]} second {y[2]} (pre-registered Q-31 cell)")
        else:
            diff += 1; print(f"DIFF {k}: first {x} second {y}")
    print(f"{len(keys) - diff - allowed} of {len(keys)} cells the same, {allowed} allowed, {diff} different")
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
