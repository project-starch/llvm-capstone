#!/usr/bin/env python3
"""fprm-grade.py <PREREG.md> <session dir>: grade the rtl lane's R-63 rounding-mode bare kit (llvm-capstone land/rtl-rm-kit
75f129f56497, capstone/tests/rtl-smoke/fp-rm-bare-2026-10-26) by its own PREREG: SR from the image table, the
bitstream-2 and fixed-RTL vectors from the backticked lines after "The bitstream-2 vector" and "The fixed-RTL vector";
the board's SR/SV recorder lines from <session>/fp-rm/uart.txt (rfix-grade.py's reader). Rules (the PREREG's): VOID
unless SR is 16 and positions 0-5 (RNE, RTZ) read exactly; CONFIRMED ON SILICON if the bitstream-2 vector is exact;
NOT ON SILICON if the fixed-RTL vector is exact; anything else a FINDING, value by value against both.
Exit 0 = CONFIRMED, 1 = NOT ON SILICON or FINDING (a reading either way), 2 = VOID or NO DATA."""
import importlib.util
import re
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("rg", Path(__file__).with_name("rfix-grade.py"))
rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)


def prereg(path):
    t = open(path).read()
    n = int(re.search(r"^\| `fp-rm\.bin` \|[^|]*\|[^|]*\| (\d+) \|", t, re.M).group(1))
    vec = {}
    for key, label in (("b2", "The bitstream-2 vector"), ("fix", "The fixed-RTL vector")):
        m = re.search(re.escape(label) + r"[^\n]*\n`([^`]+)`", t)
        vec[key] = [int(x, 16) for x in m.group(1).split()]
        if len(vec[key]) != n:
            print(f"FPRM PREREG: {label} has {len(vec[key])} values, SR {n}"); sys.exit(2)
    return n, vec


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    n, vec = prereg(sys.argv[1])
    r = rg.read(sys.argv[2], "fp-rm")
    if r is None:
        print("FPRM fp-rm NO DATA"); sys.exit(2)
    sr, sv = r
    print(f"FPRM fp-rm read: SR {sr} :: {' '.join(f'{v:#x}' for v in sv)}")
    print(f"FPRM fp-rm want (bitstream 2): SR {n} :: {' '.join(f'{v:#x}' for v in vec['b2'])}")
    if not (len(sr) == 1 and int(sr[0], 16) == n and len(sv) == n and sv[:6] == vec["b2"][:6]):
        print("FPRM VOID: SR is not 16, or RNE/RTZ (positions 0-5) do not read exactly"); sys.exit(2)
    if sv == vec["b2"]:
        print("FPRM R-63 CONFIRMED ON SILICON: RDN and RUP exchanged and RMM truncating, exactly as simulated"); sys.exit(0)
    if sv == vec["fix"]:
        print("FPRM NOT ON SILICON: the fixed-RTL vector (directed modes correctly rounded), a finding about the build"); sys.exit(1)
    d = [f"{i}: read {a:#x} b2 {b:#x} fixed {c:#x}" for i, (a, b, c) in enumerate(zip(sv, vec["b2"], vec["fix"])) if a != b]
    print(f"FPRM FINDING: {'; '.join(d)}"); sys.exit(1)


if __name__ == "__main__":
    main()
