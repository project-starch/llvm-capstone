#!/usr/bin/env python3
"""Compare a control program's EXPECT lines with slotcensus's JSON lines.

usage: check-control.py EXPECT_FILE CENSUS_JSONL
Counts the program can only add to (peak_live, allocs, frees, end_live: libc's own allocations count too) must lie in
[expected, expected + SLACK]; the realloc classifications and the fork flag must match exactly. EXPECT lines are the
parent's (forked 0), EXPECT_CHILD lines the fork child's (forked 1). Exit 0 only if every expected line has a census
line and every value holds; no census line at all is a failure, never a pass."""
import json
import sys

SLACK = 64
EXACT = {"realloc_moved", "realloc_inplace", "forked"}


def main():
    expect_path, census_path = sys.argv[1:3]
    want = {"EXPECT": {}, "EXPECT_CHILD": {}}
    for line in open(expect_path):
        parts = line.split()
        if len(parts) == 3 and parts[0] in want:
            want[parts[0]][parts[1]] = int(parts[2])
    try:
        rows = [json.loads(l) for l in open(census_path) if l.strip()]
    except FileNotFoundError:
        rows = []
    ok = True
    for who, fields in want.items():
        if not fields:
            continue
        forked = 1 if who == "EXPECT_CHILD" else 0
        mine = [r for r in rows if r["forked"] == forked]
        if len(mine) != 1:
            print(f"FAIL {who}: {len(mine)} census lines with forked={forked} (want exactly 1)")
            ok = False
            continue
        r = mine[0]
        for field, value in fields.items():
            got = r.get(field)
            good = got == value if field in EXACT else got is not None and value <= got <= value + SLACK
            print(f"{'PASS' if good else 'FAIL'} {who} {field}: census {got}, expected {value}"
                  f"{'' if field in EXACT else f' (+{SLACK})'}")
            ok &= good
    if not any(want.values()):
        print("FAIL: no EXPECT lines")
        ok = False
    print("CONTROL PASS" if ok else "CONTROL FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
