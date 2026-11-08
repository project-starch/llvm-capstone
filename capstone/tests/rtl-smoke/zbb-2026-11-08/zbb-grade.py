#!/usr/bin/env python3
"""zbb-grade.py <run.log.iss> <expected.txt>: the zbb-arm's CAPPRINT values (the Reg[] line after each committed
CAPPRINT), in order, against the host-computed expectations. No data = exit 2; any mismatch or count difference = 1."""
import sys, re
iss, exp = sys.argv[1], sys.argv[2]
want = [l.split() for l in open(exp) if l.strip()]
# The print line ("[Cycle n] Reg[10]: v") comes between the CAPPRINT's write-back and its commit, so each committed
# CAPPRINT takes the last Reg value seen since the previous committed CAPPRINT.
got = []; last = None
for line in open(iss, errors="replace"):
    m = re.search(r"Reg\[\d+\]: ([0-9a-f]+)", line)
    if m: last = int(m.group(1), 16); continue
    if "op=CAPPRINT" in line and "SUP commit" in line:
        if last is None: print("CAPPRINT committed with no print line before it"); sys.exit(1)
        got.append(last); last = None
if not got: print("NO DATA: no CAPPRINT values in", iss); sys.exit(2)
bad = 0
for i, w in enumerate(want):
    v = int(w[-1], 16)
    if i >= len(got): print(f"MISSING from case {i}: {' '.join(w)}"); bad += 1; break
    if got[i] != v:
        bad += 1
        if bad <= 20: print(f"MISMATCH case {i}: {' '.join(w[:-1])} want {v:#x} got {got[i]:#x}")
print(f"{len(got)} values read, {len(want)} expected, {bad} bad")
sys.exit(0 if bad == 0 and len(got) == len(want) else 1)
