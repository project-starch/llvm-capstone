#!/usr/bin/env python3
"""r61a-grade.py <image> <uart.txt>: grade one R-61 ARMING bare boot by the RTL lane's PREREG (capstone/tests/rtl-smoke/
r61-arming-bare-2026-10-24/PREREG.md at llvm-capstone land/rtl-r61-arming 6910f80d4903), r61-grade.py's shape.
- r61a-control: PASS when SR = 36 and the 36 values are the control vector (both trees); else VOID (the session).
- r61a-stale0 (no LDC before the legal load): PASS when SR = 19 and the normal-exit vector (both trees); else VOID FOR THE
  ARMING QUESTION (the session stops: a 15 in stale3/4 would then not be attributable to the ldc).
- r61a-stale3 (non-consuming ldc, read-only page) / r61a-stale4 (writable page, D clear): VOID unless SR = 22; CONFIRMED
  on the c721c1a2a vector; NOT-REPRODUCED on the fixed tree's (0311ca24a); any other vector a FINDING by position.
Exit 0 = PASS / CONFIRMED, 1 = NOT-REPRODUCED or FINDING, 2 = VOID or NO DATA."""
import re
import sys

H = lambda s: [int(x, 16) for x in s.split()]
CONTROL = H("0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0xd 0x0 0x0 0x5 0x99 0x0 0x5 0xf "
            "0x0 0x0 0x0 0x5 0xf 0x0 0x5 0x0 0x5 0x1c 0x0 0x0 0x5 0x0")
STALE0 = H("0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0x2 0x18 0x420590db 0x1234 0x0")
BUG = H("0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0x2 0x0 0x5 0xf 0x4 0x80006008 0x0 0x0")
FIXED = H("0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0x2 0x0 0x5 0x2 0x18 0x420590db 0x1234 0x0")
IMAGES = ("r61a-control", "r61a-stale0", "r61a-stale3", "r61a-stale4")


def grade(img, text):
    sr = re.findall(r"SR ([0-9A-Fa-f]{16})", text)
    sv = [int(v, 16) for v in re.findall(r"SV ([0-9A-Fa-f]{16})", text)]
    vals = " ".join(f"{v:#x}" for v in sv)
    one = len(sr) == 1
    if img == "r61a-control":
        return (0, f"PASS (control): {vals}") if one and int(sr[0], 16) == 36 and sv == CONTROL else (2, f"VOID: control SR {sr} :: {vals}")
    if img == "r61a-stale0":
        return (0, f"PASS (normal exit): {vals}") if one and int(sr[0], 16) == 19 and sv == STALE0 else \
            (2, f"VOID FOR THE ARMING QUESTION: stale0 SR {sr} :: {vals}")
    if not one or int(sr[0], 16) != 22 or len(sv) != 22:
        return 2, f"VOID: SR {sr}, {len(sv)} values (want one SR = 22 and 22 values)"
    if sv == BUG:
        return 0, f"CONFIRMED (the c721c1a2a vector): {vals}"
    if sv == FIXED:
        return 1, f"NOT-REPRODUCED (the fixed tree's vector on c721c1a2a): {vals}"
    d = [f"{i}: read {v:#x} c721c1a2a {b:#x} fixed {f:#x}" for i, (v, b, f) in enumerate(zip(sv, BUG, FIXED)) if v != b]
    return 1, f"FINDING: {vals} :: differs from the c721c1a2a vector at " + "; ".join(d)


def main():
    # --expect fixed (2026-10-24, for a bitstream that carries the R-61 fix): the fixed tree's vector is the PASS and the
    # c721c1a2a vector the FINDING; the controls are graded as before.
    expect = "bug"
    if "--expect" in sys.argv:
        i = sys.argv.index("--expect"); expect = sys.argv[i + 1]; del sys.argv[i:i + 2]
    if expect not in ("bug", "fixed"):
        print(f"NO DATA: unknown --expect {expect}"); sys.exit(2)
    if len(sys.argv) != 3 or sys.argv[1] not in IMAGES:
        print(__doc__); sys.exit(2)
    try:
        text = open(sys.argv[2], errors="replace").read()
    except FileNotFoundError:
        print(f"R61A {sys.argv[1]} NO DATA: no uart.txt"); sys.exit(2)
    rc, msg = grade(sys.argv[1], text)
    if expect == "fixed" and msg.startswith("CONFIRMED"):
        rc, msg = 1, "FINDING (expected the fix): R-61 still present: " + msg
    elif expect == "fixed" and msg.startswith("NOT-REPRODUCED"):
        rc, msg = 0, "FIXED (the fixed tree's vector): " + msg.split(": ", 1)[1]
    print(f"R61A {sys.argv[1]} {msg}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
