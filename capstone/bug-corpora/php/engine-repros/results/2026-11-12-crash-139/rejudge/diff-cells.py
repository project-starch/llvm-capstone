#!/usr/bin/env python3
"""diff-cells.py OLD_RECORD NEW_RECORD [--old-layout tree|merged] : every cell of the two records, both scorings.
A record is bundle-<arm>/ (default) and sensitivity/bundle-<arm>/. Prints each cell whose verdict or reason
differs, each cell where any other field of the row differs, and the counts over all cells."""
import json, sys
from pathlib import Path
old, new = Path(sys.argv[1]), Path(sys.argv[2])
ARMS = ("php-real-stock", "virtual-malloc", "virtual-nested-pools", "virtual-nested-exact")
def rows(d):
    return {json.loads(l)["observation"]["case"]: json.loads(l) for l in (d / "verdicts.jsonl").read_text().splitlines()}
total = same = moved = other = 0
for sub, name in (("", "default"), ("sensitivity/", "sensitivity")):
    for arm in ARMS:
        a, b = rows(old / f"{sub}bundle-{arm}"), rows(new / f"{sub}bundle-{arm}")
        assert set(a) == set(b), (arm, set(a) ^ set(b))
        for case in sorted(a):
            total += 1
            x, y = a[case], b[case]
            if (x["verdict"], x["reason"]) != (y["verdict"], y["reason"]):
                moved += 1
                print(f"MOVED {name:11s} {arm:22s} {case}: {x['verdict']}{':' + x['reason'] if x['reason'] else ''} -> "
                      f"{y['verdict']}{':' + y['reason'] if y['reason'] else ''}")
            elif x != y:
                other += 1
                print(f"same verdict, row differs: {name} {arm} {case}")
            else:
                same += 1
print(f"cells: {total}; row byte-identical: {same}; verdict or reason moved: {moved}; same verdict but another field differs: {other}")
