#!/usr/bin/env python3
"""Bugs caught out of ALL of each program's bugs, from a board silicon tally (records/*-silicon-tally.tsv).

Two columns per program:
  malloc  -- Sublet only in the system allocator (the vp2-malloc arm; the program's inner allocator stock);
  ported  -- the program's own inner allocator ported to derive and revoke its children
             (vp2-nested-pools, or vp2-pg-pools for PostgreSQL).
A corpus with no inner allocator on its path counts the same in both columns.

Every bar is out of all the program's bugs. A NO-READING cell (the run stopped before its bug) is neither caught nor
missed and stays in the denominator, EXCEPT the rows the lead put out of the denominator, listed in OUT with their
count; each OUT count is checked against the tally, so a changed row cannot slip through silently.
Juliet is reported on its own lines, not in the seven-program total. The mruby and perl release-differential rows
carry no catch counts and are not used.

Exit status: 0 and the table on stdout; 1 if any arm is missing or duplicated, or any count disagrees.
"""
import csv
import sys

# program -> (malloc arms, ported arms); an arm in both lists counts in both columns.
PROGRAMS = [
    ("FFmpeg", ["fca", "fph", "fpt", "fso", "pb9", "plane-malloc"],
               ["fcs", "fph", "fpt", "fso", "pb10", "plane-nested"]),
    ("tshark", ["wph", "wpt", "wm0"], ["wph", "wpt", "wm1q15"]),
    ("memcached", ["mph", "mpt", "pb12"], ["mph", "mpt", "pb13q15"]),
    ("CPython", ["pym0", "cpython-lt0"], ["pym1", "cpython-lt1"]),
    ("PostgreSQL", ["pgc", "pgm0", "sql0"], ["pgc", "pgm1q15", "pg-pools"]),
    ("SQLite", ["sq0", "capi"], ["sq1q15", "capi"]),
    ("Lua, mruby shims", ["xlang-repro", "xlang-lua-cdp"], ["xlang-repro", "xlang-lua-cdp"]),
]
# Out of the denominator by the lead's decisions: arm -> NO-READING cells removed.
OUT = {
    "capi": 7,          # SQLite capi: null dereferences and rows with nothing to catch
    "xlang-repro": 11,  # the mruby shim rows
}
# An arm read on both boards for the same bugs: asserted equal, counted once.
TWINS = {"pgm1": "pgm1q15"}
JULIET = ["juliet-temporal", "juliet-spatial"]


def die(msg):
    print(f"catch-table: {msg}", file=sys.stderr)
    sys.exit(1)


def main(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["arm_tag"] in rows:
                die(f"duplicate arm {r['arm_tag']} in {path}")
            rows[r["arm_tag"]] = r

    def counts(tag):
        if tag not in rows:
            die(f"arm {tag} not in {path}")
        r = rows[tag]
        try:
            c, m, n, cells = (int(r[k]) for k in ("CAUGHT", "MISSED", "NO-READING", "cells"))
        except ValueError:
            die(f"{tag}: a count column is empty")
        if c + m + n != cells:
            die(f"{tag}: {c}+{m}+{n} != {cells} cells")
        out = OUT.get(tag, 0)
        if n < out:
            die(f"{tag}: {n} NO-READING, but {out} declared out of the denominator")
        return c, m, n - out

    for a, b in TWINS.items():
        if counts(a) != counts(b):
            die(f"twin arms {a} {counts(a)} and {b} {counts(b)} differ")

    print(f"source: {path}")
    print(f"{'program':18} {'bugs':>4}  {'malloc c/m/n':>14}  {'ported c/m/n':>14}")
    tot = [[0, 0, 0], [0, 0, 0]]
    for name, mal, por in PROGRAMS:
        cols = []
        for i, arms in enumerate((mal, por)):
            s = [0, 0, 0]
            for t in arms:
                s = [x + y for x, y in zip(s, counts(t))]
            cols.append(s)
            tot[i] = [x + y for x, y in zip(tot[i], s)]
        if sum(cols[0]) != sum(cols[1]):
            die(f"{name}: malloc column has {sum(cols[0])} bugs, ported {sum(cols[1])}")
        print(f"{name:18} {sum(cols[0]):4}  {'%d/%d/%d' % tuple(cols[0]):>14}  {'%d/%d/%d' % tuple(cols[1]):>14}")
    total = sum(tot[0])
    print(f"{'All seven':18} {total:4}  {'%d/%d/%d' % tuple(tot[0]):>14}  {'%d/%d/%d' % tuple(tot[1]):>14}")
    print(f"caught: malloc {tot[0][0]}/{total} ({100 * tot[0][0] / total:.1f} %), "
          f"ported {tot[1][0]}/{total} ({100 * tot[1][0] / total:.1f} %)")
    for t in JULIET:
        c, m, n = counts(t)
        print(f"{t}: caught {c} / {c + m + n} (missed {m}, no reading {n})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        die("usage: catch-table.py <silicon-tally.tsv>")
    main(sys.argv[1])
