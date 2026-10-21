#!/usr/bin/env python3
"""s23-grade.py <uart.txt> [--expect pre|batch]: grade one bare s23-cpmp-rearm boot by the RTL lane's rule
(capstone/tests/rtl-smoke/s23-bare-2026-10-21/PREREG.md at 0c215bf0983f).

The report is "SR <16 hex>" (the count of SV values) and "SV <16 hex>" lines, in order: a0, the marker 0x5a23,
Q1..Q13d (20 deltas, 0 ALLOWED / 1 DENIED), the last cause, the total trap count, the end marker 0x55.
- VOID unless SR = 25, there are 25 SV values, the marker and end marker are in place, and Q1 = 0.
- VOID (instrument moved) unless the eight true-deny and the six control probes equal the vector.
- Then the six re-arm probes and the total: pre-batch reads all 0 and total 8; batch reads all 1 and total 14.
  Any other mix is a FINDING. A re-arm probe reading 1 where pre-batch is expected is a FINDING to surface.
Exit 0 = reads as expected, 1 = FINDING, 2 = VOID or NO DATA."""
import re
import sys

NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11a", "Q11b", "Q11c", "Q11d",
         "Q12a", "Q12b", "Q13a", "Q13b", "Q13c", "Q13d"]
PRE = [0x0, 0x5a23, 0x0, 0x0, 0x1, 0x1, 0x0, 0x0, 0x1, 0x0, 0x1, 0x0, 0x0, 0x0, 0x1, 0x1, 0x0, 0x0, 0x0, 0x1, 0x1, 0x0,
       0x5, 0x8, 0x55]
BATCH = [0x0, 0x5a23, 0x0, 0x0, 0x1, 0x1, 0x1, 0x0, 0x1, 0x1, 0x1, 0x1, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x0, 0x1, 0x1, 0x1,
         0x5, 0xe, 0x55]
REARM = ["Q5", "Q8", "Q10", "Q12a", "Q12b", "Q13d"]


def grade(text, expect):
    sr = re.findall(r"SR ([0-9A-Fa-f]{16})", text)
    sv = [int(v, 16) for v in re.findall(r"SV ([0-9A-Fa-f]{16})", text)]
    if len(sr) != 1 or int(sr[0], 16) != 25:
        return 2, f"VOID: SR {sr} (want one SR = 25)"
    if len(sv) != 25 or sv[1] != 0x5a23 or sv[24] != 0x55:
        return 2, f"VOID: {len(sv)} SV values, marker {sv[1:2]}, end {sv[24:25]} (want 25, 0x5a23, 0x55)"
    probes = dict(zip(NAMES, sv[2:22]))
    if probes["Q1"] != 0:
        return 2, "VOID: Q1 reads 1 (the instrument control)"
    fixed = [n for n in NAMES if n not in REARM]
    moved = [n for n in fixed if probes[n] != PRE[2 + NAMES.index(n)]]
    if moved or sv[22] != 0x5 or sv[0] != 0x0:
        return 2, f"VOID (instrument moved): true-deny/control probes {moved}, last cause {sv[22]:#x}, a0 {sv[0]:#x}"
    flips = {n: probes[n] for n in REARM}
    total = sv[23]
    if all(v == 0 for v in flips.values()) and total == 8:
        reading = "pre"
    elif all(v == 1 for v in flips.values()) and total == 14:
        reading = "batch"
    else:
        return 1, f"FINDING: re-arm probes {flips}, total {total} (neither pre-batch 0/8 nor batch 1/14)"
    if reading != expect:
        return 1, f"FINDING: reads {reading} (re-arm {flips}, total {total}) where {expect} is expected"
    return 0, f"AS EXPECTED ({expect}): SR 25, Q1 0, every true-deny and control probe as the vector, re-arm {flips}, total {total}"


def main():
    path = sys.argv[1]
    expect = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--expect" else "pre"
    try:
        text = open(path, errors="replace").read()
    except FileNotFoundError:
        print(f"NO DATA: {path} does not exist")
        sys.exit(2)
    rc, msg = grade(text, expect)
    print("S23", msg)
    sys.exit(rc)


if __name__ == "__main__":
    main()
