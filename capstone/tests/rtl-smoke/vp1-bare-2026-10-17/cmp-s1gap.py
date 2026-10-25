#!/usr/bin/env python3
"""cmp-s1gap.py <ref-dir> <new-dir>: s1gap against its d93a8d66b record, by the test's own reading layout
(s1-revoke-gap-2026-10-14/PREREG.md: for k = 1..4 the gap cycles, the revoke cycles, the revoke cause; then the base's
lcc validity, total traps, GAP, 4). The cycle counts (positions 0,1,3,4,6,7,9,10) vary run to run on d93a8d66b itself
(three runs, 2026-10-16), so they are REPORTED with the new value; every other position, and every reading of the
control arm, must be identical. Exit 0 only if every arm is present with the reference's length and every functional
position matches; 1 otherwise; 2 on no data."""
import re, sys, pathlib
TIMED = {0, 1, 3, 4, 6, 7, 9, 10}
ref, new = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
arms = sorted(p.name[5:-4] for p in ref.glob("runs-*.log"))
if not arms: print(f"ERROR: no runs-*.log under {ref}"); sys.exit(2)
def vec(p):
    return [int(v, 16) for v in re.findall(r"reading +\d+: (0x[0-9a-fA-F]+)", p.read_text(errors="replace"))] if p.exists() else None
bad = 0
for a in arms:
    r, n = vec(ref / f"runs-{a}.log"), vec(new / f"runs-{a}.log")
    if not r: print(f"ERROR {a}: no reference readings"); bad += 1; continue
    if not n: print(f"MISSING {a}"); bad += 1; continue
    if len(r) != len(n): print(f"DIFF {a}: length {len(n)} against {len(r)}"); bad += 1; continue
    timed = TIMED if a.startswith("s1gap") else set()
    fdiff = [i for i in range(len(r)) if i not in timed and r[i] != n[i]]
    tdiff = [i for i in sorted(timed) if r[i] != n[i]]
    if fdiff: bad += 1; print(f"DIFF {a}: functional positions {fdiff}: ref {[hex(r[i]) for i in fdiff]} new {[hex(n[i]) for i in fdiff]}")
    else: print(f"SAME {a} functional ({len(r) - len(timed)})" + (f"; timed positions differing {tdiff}: ref {[hex(r[i]) for i in tdiff]} new {[hex(n[i]) for i in tdiff]}" if tdiff else ""))
print(f"{len(arms)} arms, {bad} with a functional difference")
sys.exit(1 if bad else 0)
