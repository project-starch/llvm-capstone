#!/usr/bin/env python3
"""fpds-b3-grade.py <session dir>: grade the rtl lane's fp-all bare image on BITSTREAM 3 (capstone-ariane e7af48457) by its
"Bitstream 3" PREREG section (llvm-capstone land/rtl-b3-prereg bc8e895d8db1, capstone/tests/rtl-smoke/
fp-divsqrt-bare-2026-10-26/PREREG.md), read from that commit with git:
- the e7af48457 vector, from that section's table row;
- the dcc6daf6e (bitstream 2) vector, from the kit's original table row;
- the board's SR/SV lines, from <session>/fp-all/uart.txt.
Rules (the PREREG's):
- VOID unless SR is 10 and position 8 is 0xc100794b211e882c;
- R-62 FIXED ON SILICON if the e7af48457 vector reads exactly;
- STILL THE DEFECT if the dcc6daf6e vector reads (check the resident bitstream first);
- anything else is a FINDING, value by value against the e7af48457 vector.
Exit 0 = FIXED, 1 = STILL THE DEFECT or FINDING, 2 = VOID or NO DATA. Written 2026-10-27."""
import re
import subprocess
import sys
from pathlib import Path

C = "bc8e895d8db1"
P = "capstone/tests/rtl-smoke/fp-divsqrt-bare-2026-10-26/PREREG.md"
LAST = 0xc100794b211e882c


def vectors():
    t = subprocess.run(["git", "-C", "/tmp/capstone/wt-gate", "show", f"{C}:{P}"], capture_output=True, text=True)
    if t.returncode:
        print(f"FPDS-B3 NO DATA: cannot read {C}:{P}"); sys.exit(2)
    b3_part = t.stdout.split("## Bitstream 3", 1)
    if len(b3_part) != 2:
        print("FPDS-B3 NO DATA: no Bitstream 3 section"); sys.exit(2)
    def row(text, ncols):
        for l in text.splitlines():
            if l.startswith("| `fp-all.bin`"):
                c = [x.strip() for x in l.split("|")[1:-1]]
                if len(c) == ncols:
                    return int(c[2 if ncols == 4 else 3]), [int(x, 16) for x in re.search(r"`([^`]+)`", c[-1]).group(1).split()]
        return None
    b3, b2 = row(b3_part[1], 4), row(b3_part[0], 5)
    if not b3 or not b2:
        print(f"FPDS-B3 NO DATA: table rows (bitstream 3 {b3 is not None}, bitstream 2 {b2 is not None})"); sys.exit(2)
    return b3, b2


def main():
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    (n3, w3), (n2, w2) = vectors()
    p = Path(sys.argv[1]) / "fp-all" / "uart.txt"
    if not p.is_file():
        print(f"FPDS-B3 NO DATA: no {p}"); sys.exit(2)
    t = p.read_text(errors="replace")
    sr = re.findall(r"SR ([0-9A-Fa-f]{16})", t); sv = [int(v, 16) for v in re.findall(r"SV ([0-9A-Fa-f]{16})", t)]
    print(f"FPDS-B3 read: SR {sr} :: {' '.join(f'{v:#x}' for v in sv)}")
    print(f"FPDS-B3 want (e7af48457): SR {n3} :: {' '.join(f'{v:#x}' for v in w3)}")
    if not (len(sr) == 1 and int(sr[0], 16) == n3 == 10 and len(sv) == n3 and sv[8] == LAST):
        print("FPDS-B3 VOID: SR is not 10 or position 8 is not the last quotient"); sys.exit(2)
    if sv == w3:
        print("FPDS-B3 R-62 FIXED ON SILICON: the e7af48457 vector exactly (0 fdiv and 0 fsqrt misses, NX set)"); sys.exit(0)
    if sv == w2:
        print("FPDS-B3 STILL THE DEFECT: the dcc6daf6e vector; check the resident bitstream first"); sys.exit(1)
    d = [f"{i}: read {a:#x} want {b:#x}" for i, (a, b) in enumerate(zip(sv, w3)) if a != b]
    print(f"FPDS-B3 FINDING: {'; '.join(d)}"); sys.exit(1)


if __name__ == "__main__":
    main()
