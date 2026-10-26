#!/usr/bin/env python3
"""fpds-grade.py <PREREG.md> <session dir>: grade the rtl lane's fdiv.d/fsqrt.d rounding bare kit (llvm-capstone
land/rtl-fp-kit 84488ed6db17, capstone/tests/rtl-smoke/fp-divsqrt-bare-2026-10-26) by its own PREREG: fp-all's SR and
vector from the table (rfix-grade.py's table reader), the board's `SR`/`SV` recorder lines from <session>/fp-all/uart.txt.
Rules (the PREREG's): VOID unless SR is 10 and position 8 is 0xc100794b211e882c; CONFIRMED ON SILICON if the vector is
exact; NOT ON SILICON if positions 0 and 1 read 0 with position 8 exact; anything else a FINDING, value by value.
Exit 0 = CONFIRMED, 1 = NOT ON SILICON or FINDING (a reading either way), 2 = VOID or NO DATA."""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("rg", Path(__file__).with_name("rfix-grade.py"))
rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)
LAST = 0xc100794b211e882c


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    n, want = rg.table(sys.argv[1])["fp-all"]
    r = rg.read(sys.argv[2], "fp-all")
    if r is None:
        print("FPDS fp-all NO DATA"); sys.exit(2)
    sr, sv = r
    vals = " ".join(f"{v:#x}" for v in sv)
    print(f"FPDS fp-all read: SR {sr} :: {vals}")
    print(f"FPDS fp-all want: SR {n} :: {' '.join(f'{v:#x}' for v in want)}")
    if not (len(sr) == 1 and int(sr[0], 16) == n and len(sv) == n and sv[8] == LAST):
        print("FPDS VOID: SR is not 10 or position 8 is not the last quotient (the loop did not finish, or not this image)"); sys.exit(2)
    if sv == want:
        print("FPDS CONFIRMED ON SILICON: the rounding defect is on the board (the vector is exact)"); sys.exit(0)
    if sv[0] == 0 and sv[1] == 0:
        print("FPDS NOT ON SILICON: 0 fdiv and 0 fsqrt mismatches on the 512 cases, position 8 exact"); sys.exit(1)
    d = [f"{i}: read {a:#x} want {b:#x}" for i, (a, b) in enumerate(zip(sv, want)) if a != b]
    print(f"FPDS FINDING: {'; '.join(d)}"); sys.exit(1)


if __name__ == "__main__":
    main()
