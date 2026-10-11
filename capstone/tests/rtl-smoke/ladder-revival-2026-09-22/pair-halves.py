#!/usr/bin/env python3
"""Pair the ladder's two halves into overhead ratios, and REFUSE to if the control fails.

Usage: pair-halves.py <baseline-result-lines> <capability-result-lines> [more capability files ...]

The baseline file is either format:
  * BARE-METAL (the instrument of record, issue I-2):
      `<rung>: BEST cycles=C instret=I (T/N passes at min instret, spread=S) retval=R`
  * the retired LINUX-userspace table (kept readable only so older captures in this
    folder can still be re-checked; its rows are floors only when clean is high).

Capability files are the driver's `rung retval oracle cycles instret correct` tables.
A rung measured in several boots keeps its MINIMUM cycle count; its instret must be
identical across boots (the domain half is deterministic) or the rung is excluded.

Gates, all of which refuse rather than warn:
  * the control rung must be present in both halves;
  * its baseline must be a floor (clean >= CLEAN_MIN);
  * its instruction ratio must be within CONTROL_TOL of 1.000 (identical code both sides);
  * its baseline CPI must be within CPI_TOL of CONTROL_BASE_CPI -- the value the bare-metal
    baseline has read since 2026-07-28, so a denominator that drifted from the July floor
    is caught before any row is printed.
Per row: a capability value that is not its oracle, a baseline retval that differs from the
capability retval, a baseline floor below CLEAN_MIN, or a missing instret excludes the row.
"""
import re, sys

CONTROL = "ctrsanity"
CONTROL_BASE_CPI = 1.2000
CPI_TOL = 0.005
CLEAN_MIN = 10
CONTROL_TOL = 0.01
NOT_BENCHMARKS = {"null", "rawhazard5", "rawhazard6", "rawhazard7"}

BARE = re.compile(r"(\w+): BEST cycles=(\d+) instret=(\d+) \((\d+)/(\d+) passes at min "
                  r"instret, spread=(\d+)\) retval=(\d+)")
LINUX = re.compile(r"(\w+)\s+(\d+)\s+(\d+|--)\s+(\d+|--)\s+(\d+|--)\s+(\d+|--)\s+(\S+)\s+(\S+)"
                   r"\s+(\S+)\s+(YES|NO)")
CAP = re.compile(r"(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+|None)\s+(YES|NO)")


def read_base(path):
    """rung -> (cycles, instret, tied, total, retval-or-None)"""
    out = {}
    for ln in open(path):
        ln = ln.strip()
        m = BARE.match(ln)
        if m:
            r, c, i, t, n, _s, rv = m.groups()
            out[r] = (int(c), int(i), int(t), int(n), int(rv))
            continue
        m = LINUX.match(ln)
        if m:
            r, _, _cc, bc, _ci, bi, clean, _sp, idem, ok = m.groups()
            if "--" in (bc, bi) or ok != "YES" or not idem.startswith("YES"):
                continue
            t, _, n = clean.partition("/")
            out[r] = (int(bc), int(bi), int(t), int(n or 0), None)
    return out


def read_caps(paths):
    """rung -> (min cycles, instret, retval, boots); None instret or wrong oracle excluded."""
    seen, bad = {}, {}
    for p in paths:
        for ln in open(p):
            m = CAP.match(ln.strip())
            if not m or m.group(1) == "rung":
                continue
            r, rv, orc, cyc, ins, ok = m.groups()
            if ok != "YES" or rv != orc:
                bad[r] = "capability value is not its oracle"; continue
            if ins == "None":
                bad[r] = "no instret recorded"; continue
            seen.setdefault(r, []).append((int(cyc), int(ins), int(rv)))
    out = {}
    for r, obs in seen.items():
        if len({o[1] for o in obs}) != 1:
            bad[r] = f"instret differs across boots {sorted({o[1] for o in obs})}"; continue
        out[r] = (min(o[0] for o in obs), obs[0][1], obs[0][2], len(obs),
                  max(o[0] for o in obs) - min(o[0] for o in obs))
    return out, bad


def main(basef, capfs):
    base = read_base(basef)
    cap, bad = read_caps(capfs)
    if not base:
        sys.exit(f"REFUSED: no baseline rows parsed from {basef} (neither known format)")
    if CONTROL not in cap or CONTROL not in base:
        sys.exit(f"REFUSED: control {CONTROL} missing from a half; no ratio is valid without it")
    bc, bi, tied, tot, _ = base[CONTROL]
    cc, ci = cap[CONTROL][:2]
    iratio, bcpi = ci / bi, bc / bi
    print(f"control {CONTROL}: clean={tied}/{tot}  instr ratio={iratio:.5f}  "
          f"baseline CPI={bcpi:.4f}  cycle ratio={cc/bc:.4f}")
    if tied < CLEAN_MIN:
        sys.exit(f"REFUSED: control baseline is not a floor (clean={tied}/{tot} < {CLEAN_MIN})")
    if abs(iratio - 1.0) > CONTROL_TOL:
        sys.exit(f"REFUSED: control instruction ratio {iratio:.4f} is outside {CONTROL_TOL:.0%} "
                 "of 1.000 on identical code; the halves are not matched")
    if abs(bcpi - CONTROL_BASE_CPI) > CPI_TOL:
        sys.exit(f"REFUSED: control baseline CPI {bcpi:.4f} is not the {CONTROL_BASE_CPI:.4f} "
                 "floor the bare-metal baseline has read since 2026-07-28; the denominator moved")
    print("control PASSED -- rows below are admissible "
          "(the control's own cycle ratio is reported, not gated)\n")
    print(f"{'rung':<18}{'cap cyc':>11}{'base cyc':>11}{'cycles':>9}{'cap ins':>11}"
          f"{'base ins':>11}{'instr':>8}{'CPI':>8}{'boots':>6}{'cap spread':>11}")
    rows = []
    for r in sorted(cap, key=lambda r: (cap[r][0] / base[r][0]) if r in base else 9e9):
        if r in NOT_BENCHMARKS:
            continue
        if r not in base:
            print(f"{r:<18}  -- no baseline row"); continue
        cc, ci, crv, boots, cspread = cap[r]
        bc, bi, tied, tot, brv = base[r]
        why = []
        if tied < CLEAN_MIN: why.append(f"baseline not a floor ({tied}/{tot})")
        if brv is not None and brv != crv: why.append(f"baseline retval {brv} != {crv}")
        tag = "   <- EXCLUDED: " + "; ".join(why) if why else ""
        ctl = "  (control)" if r.startswith(CONTROL) else ""
        print(f"{r:<18}{cc:>11,}{bc:>11,}{cc/bc:>8.3f}x{ci:>11,}{bi:>11,}{ci/bi:>8.3f}"
              f"{(cc/ci)/(bc/bi):>8.3f}{boots:>6}{cspread:>11}{ctl}{tag}")
        if not why: rows.append(r)
    for r, why in sorted(bad.items()):
        if r not in NOT_BENCHMARKS:
            print(f"{r:<18}  -- EXCLUDED: {why}")
    print(f"\n{len(rows)} admissible rows")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
