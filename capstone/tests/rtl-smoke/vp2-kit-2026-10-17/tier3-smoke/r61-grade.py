#!/usr/bin/env python3
"""r61-grade.py <image> <uart.txt>: grade one R-61 bare boot by the RTL lane's PREREG (capstone/tests/rtl-smoke/
r61-stale-bare-2026-10-24/PREREG.md at llvm-capstone land/rtl-r61-kit 61faf96f5863). The recorder lines are `SR <16 hex>`
(the count) and `SV <16 hex>` (the values), as va-alloc-grade.py reads them.
- r61-control: PASS when SR = 36 and all 36 values equal the control vector (identical on both trees); else VOID (the
  whole session is void).
- r61-stale2 / r61-stale: VOID unless SR is the table's (23 / 22). CONFIRMED when the values equal the c721c1a2a
  vector; NOT-REPRODUCED when they equal the fixed tree's (dcc6daf6e) vector; any other vector is a FINDING, printed
  position by position against both.
Exit 0 = PASS / CONFIRMED, 1 = NOT-REPRODUCED or FINDING, 2 = VOID or NO DATA."""
import re
import sys

H = lambda s: [int(x, 16) for x in s.split()]
CONTROL = H("0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0xd 0x0 0x0 0x5 0x99 0x0 0x5 0xf "
            "0x0 0x0 0x0 0x5 0xf 0x0 0x5 0x0 0x5 0x1c 0x0 0x0 0x5 0x0")
VEC = {
    "r61-stale2": (23, H("0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0xf 0x0 0x0 0x5 0xf 0x4 "
                         "0x80006008 0x0 0x0"),
                   H("0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0xf 0x0 0x0 0x5 0x2 0x18 "
                     "0x420590db 0x1234 0x0")),
    "r61-stale": (22, H("0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0xf 0x0 0x0 0x5 0xf 0x0 "
                        "0x0 0x0"),
                  H("0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0xf 0x0 0x0 0x5 0xd 0x0 "
                    "0x0 0x0")),
}


def grade(img, text):
    sr = re.findall(r"SR ([0-9A-Fa-f]{16})", text)
    sv = [int(v, 16) for v in re.findall(r"SV ([0-9A-Fa-f]{16})", text)]
    vals = " ".join(f"{v:#x}" for v in sv)
    if img == "r61-control":
        if len(sr) == 1 and int(sr[0], 16) == 36 and sv == CONTROL:
            return 0, f"PASS (control): {vals}"
        return 2, f"VOID: control SR {sr}, {len(sv)} values, differing at {[i for i, (a, b) in enumerate(zip(sv, CONTROL)) if a != b]} :: {vals}"
    n, bug, fixed = VEC[img]
    if len(sr) != 1 or int(sr[0], 16) != n or len(sv) != n:
        return 2, f"VOID: SR {sr}, {len(sv)} values (want one SR = {n} and {n} values)"
    if sv == bug:
        return 0, f"CONFIRMED (the c721c1a2a vector): {vals}"
    if sv == fixed:
        return 1, f"NOT-REPRODUCED (the fixed tree's vector on c721c1a2a): {vals}"
    d = [f"{i}: read {v:#x} c721c1a2a {b:#x} fixed {f:#x}" for i, (v, b, f) in enumerate(zip(sv, bug, fixed)) if v != b]
    return 1, f"FINDING: {vals} :: differs from the c721c1a2a vector at " + "; ".join(d)


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("r61-control", *VEC):
        print(__doc__); sys.exit(2)
    try:
        text = open(sys.argv[2], errors="replace").read()
    except FileNotFoundError:
        print(f"R61 {sys.argv[1]} NO DATA: no uart.txt"); sys.exit(2)
    rc, msg = grade(sys.argv[1], text)
    print(f"R61 {sys.argv[1]} {msg}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
