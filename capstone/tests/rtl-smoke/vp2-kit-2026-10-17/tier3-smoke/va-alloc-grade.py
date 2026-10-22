#!/usr/bin/env python3
"""va-alloc-grade.py <uart.txt>: grade one bare va-alloc-csret boot by the RTL lane's rule
(capstone/tests/rtl-smoke/va-alloc-bare-2026-10-23/PREREG.md at dev 68679d2982ad).
VOID unless there is one SR = 20, 20 SV values, and position 0 = 0. PASS when all 20 equal the batch vector (c721c1a2a).
PRE-BATCH when all 20 equal tree A's vector (the bitstream in service is not the batch). Any other vector is a FINDING,
printed position by position against both. Exit 0 = PASS, 1 = PRE-BATCH or FINDING, 2 = VOID or NO DATA."""
import re
import sys

BATCH = [0x0, 0x10004, 0x0, 0x0, 0x20004, 0x1, 0x1, 0x0, 0x0, 0x0, 0x4, 0x3, 0x0, 0x0, 0x2, 0x3, 0x4, 0x1, 0x2, 0x0]
PRE = [0x0, 0x4, 0x0, 0x0, 0x7, 0x1, 0x1, 0x0, 0x0, 0x0, 0x4, 0x2, 0x0, 0x0, 0x0, 0x2, 0x4, 0x1, 0x0, 0x0]


def grade(text):
    sr = re.findall(r"SR ([0-9A-Fa-f]{16})", text)
    sv = [int(v, 16) for v in re.findall(r"SV ([0-9A-Fa-f]{16})", text)]
    if len(sr) != 1 or int(sr[0], 16) != 20 or len(sv) != 20:
        return 2, f"VOID: SR {sr}, {len(sv)} SV values (want one SR = 20 and 20 values)"
    if sv[0] != 0:
        return 2, f"VOID: position 0 reads {sv[0]:#x} (want 0)"
    vals = " ".join(f"{v:#x}" for v in sv)
    if sv == BATCH:
        return 0, f"PASS (batch vector): {vals}"
    if sv == PRE:
        return 1, f"PRE-BATCH (tree A's vector: the bitstream in service is not the batch): {vals}"
    diff = [f"{i}: read {v:#x} batch {b:#x} pre {p:#x}" for i, (v, b, p) in enumerate(zip(sv, BATCH, PRE)) if v != b]
    return 1, f"FINDING: {vals} :: differs from the batch at " + "; ".join(diff)


def main():
    try:
        text = open(sys.argv[1], errors="replace").read()
    except (IndexError, FileNotFoundError):
        print("NO DATA: no uart.txt"); sys.exit(2)
    rc, msg = grade(text)
    print("VA-ALLOC", msg)
    sys.exit(rc)


if __name__ == "__main__":
    main()
