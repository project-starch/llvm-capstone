#!/usr/bin/env python3
"""jrem-classify.py <obs-census.cmp> <obs-seeded.cmp>: class every DIFF line juliet-obs-compare.py wrote for one Juliet
remainder part. It replaces the runner's inline classifier (tier3-juliet-rem-b2.sh), whose labels were found wrong on
part01 (2026-10-26):
  - it took the key up to the FIRST colon, so `controls:ctl-uaf` became `controls` and the uaf-control class never fired;
  - it labelled any `_12.` key a rand-driven draw, including in the SEEDED run, whose build pins the draw, so a seeded
    difference can never be one.
The grade itself (cell equality, no tolerance) is the runner's and is unchanged; this only names each difference.
Classes:
  uaf-control       controls:ctl-uaf, where every differing step reads cause 13 for the reference's 25 at the same site
  rand _12 draw     a `_12.` case in the CENSUS run only
  CWE170 class      the case name contains CWE170 (addendum de7d470759c3)
  OTHER             anything else; a step at cause 4 (misaligned) is named so
Prints one CLASS line per difference and one COUNT line per class. Exit 0 when both files parsed, 2 on no data."""
import ast
import re
import sys
from collections import Counter


def parse(path):
    try:
        text = open(path).read()
    except FileNotFoundError:
        print(f"NO DATA: {path} does not exist"); sys.exit(2)
    if not re.search(r"^\d+ of \d+ cells the same", text, re.M):
        print(f"NO DATA: {path} has no summary line"); sys.exit(2)
    out = []
    for line in text.splitlines():
        m = re.match(r"DIFF (.+?): reference (\[.*\]|None) board (\[.*\]|None)$", line)
        if line.startswith("DIFF ") and not m:
            print(f"NO DATA: unparsed DIFF line in {path}: {line[:120]}"); sys.exit(2)
        if m:
            out.append((m.group(1), ast.literal_eval(m.group(2)), ast.literal_eval(m.group(3))))
    return out


def classify(key, ref, board, seeded):
    if key == "controls:ctl-uaf" and ref and board and len(ref) == len(board):
        moved = [(a, b) for a, b in zip(ref, board) if a != b]
        if moved and all(a[1] == 25 and b[1] == 13 and a[0] == b[0] and a[2] == b[2] and a[3] == b[3] for a, b in moved):
            return "uaf-control", f"{len(moved)} of {len(ref)} steps cause 13"
    if "CWE170" in key:
        return "CWE170 class", ""
    if "_12." in key and not seeded:
        return "rand _12 draw", ""
    mis = [b for b in (board or []) if b[1] == 4]
    return "OTHER", (f"board cause 4 (misaligned) at {', '.join(b[2] for b in mis)}" if mis else "")


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    n = Counter()
    for path, seeded in ((sys.argv[1], False), (sys.argv[2], True)):
        for key, ref, board in parse(path):
            c, note = classify(key, ref, board, seeded)
            n[c] += 1
            r = "; ".join(f"{x[0][0]} {x[0][1]} cause={x[1]} {x[2]}" for x in (ref or [])[:1])
            b = "; ".join(f"{x[0][0]} {x[0][1]} cause={x[1]} {x[2]}" for x in (board or [])[:1])
            print(f"CLASS [{c}] {'seeded' if seeded else 'census'} {key} :: ref {r} :: board {b}" + (f" :: {note}" if note else ""))
    for c in ("uaf-control", "rand _12 draw", "CWE170 class", "OTHER"):
        print(f"COUNT {c} {n[c]}")
    sys.exit(0)


if __name__ == "__main__":
    main()
