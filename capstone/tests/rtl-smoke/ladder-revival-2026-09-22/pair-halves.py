#!/usr/bin/env python3
"""Pair the ladder's two halves into overhead ratios, and REFUSE to if the control fails.

Usage: pair-halves.py <capability-result-lines> <baseline-result-lines>

The capability half is a domain measurement: no paging, no interrupts, and it
reproduces to the instruction across boots. The baseline half runs in Linux
userspace, so its per-pass counters take timer interrupts; the runner reports the
minimum-instret pass (`best_*`) plus `clean` = passes tied at that minimum.

`clean` is the gate, not decoration. clean=1/N means the floor was never reached
and that row is not a denominator. The control rung must additionally read an
instruction ratio within 1% of 1.000, because both its halves run identical code
-- if it does not, nothing else here is trustworthy and no ratio is emitted.
"""
import re, sys

CONTROL = "ctrsanitys"
CLEAN_MIN = 10          # passes tied at the minimum, out of the warm passes
CONTROL_TOL = 0.01

def read_cap(path):
    out = {}
    for ln in open(path):
        m = re.match(r"(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+|None)\s+(YES|NO)", ln.strip())
        if m and m.group(6) == "YES" and m.group(5) != "None":
            out[m.group(1)] = (int(m.group(4)), int(m.group(5)))   # cycles, instret
    return out

def read_base(path):
    out = {}
    for ln in open(path):
        m = re.match(r"(\w+)\s+(\d+)\s+(\d+|--)\s+(\d+|--)\s+(\d+|--)\s+(\d+|--)\s+(\S+)\s+(\S+)\s+(\S+)\s+(YES|NO)", ln.strip())
        if not m: continue
        rung, _, cold_c, best_c, cold_i, best_i, clean, spread, idem, ok = m.groups()
        if "--" in (best_c, best_i) or ok != "YES": continue
        tied, _, tot = clean.partition("/")
        out[rung] = (int(best_c), int(best_i), int(tied), int(tot or 0), idem)
    return out

def main(capf, basef):
    cap, base = read_cap(capf), read_base(basef)
    if CONTROL not in cap or CONTROL not in base:
        sys.exit(f"REFUSED: control {CONTROL} missing from a half; no ratio is valid without it")
    bc, bi, tied, tot, idem = base[CONTROL]
    cc, ci = cap[CONTROL]
    ratio = ci / bi
    print(f"control {CONTROL}: clean={tied}/{tot}  instr ratio={ratio:.4f}  cycle ratio={cc/bc:.4f}")
    if tied < CLEAN_MIN:
        sys.exit(f"REFUSED: control floor not reached (clean={tied}/{tot} < {CLEAN_MIN}); "
                 "the baseline half cannot measure a control, so no overhead row is emitted")
    if abs(ratio - 1.0) > CONTROL_TOL:
        sys.exit(f"REFUSED: control instruction ratio {ratio:.4f} is outside "
                 f"{CONTROL_TOL:.0%} of 1.000 on identical code; the halves are not matched")
    print(f"\ncontrol PASSED -- ratios below are admissible\n")
    print(f"{'rung':<20}{'cap cyc':>10}{'base cyc':>10}{'cyc ratio':>11}"
          f"{'cap ins':>10}{'base ins':>10}{'ins ratio':>11}{'clean':>8}")
    for r in sorted(cap):
        if r not in base or r == CONTROL: continue
        bc, bi, tied, tot, idem = base[r]
        cc, ci = cap[r]
        note = "" if tied >= CLEAN_MIN and idem.startswith("YES") else "   <- excluded"
        print(f"{r:<20}{cc:>10,}{bc:>10,}{cc/bc:>11.3f}{ci:>10,}{bi:>10,}"
              f"{ci/bi:>11.3f}{f'{tied}/{tot}':>8}{note}")

if __name__ == "__main__":
    main(*sys.argv[1:3])
