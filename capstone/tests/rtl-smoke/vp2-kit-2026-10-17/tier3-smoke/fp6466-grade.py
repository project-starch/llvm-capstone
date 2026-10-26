#!/usr/bin/env python3
"""fp6466-grade.py <PREREG.md> <session dir>: grade the rtl lane's R-64/R-65/R-66 bare kit (llvm-capstone land/rtl-r6466-kit
5adf7a954946, capstone/tests/rtl-smoke/fp-r64-r66-bare-2026-10-27) on bitstream 3 by its own PREREG.
- Per image: SR from the image table, and the two vectors from the backticked lines after "Bitstream 3 (" and
  "Fixed core (" in its "### <image>" section. Each vector must have SR values, or the PREREG is unreadable (exit 2).
- The board's SR/SV recorder lines come from <session>/<image>/uart.txt (rfix-grade.py's reader).
Rules (the PREREG's):
- VOID unless SR matches;
- CONFIRMED ON SILICON if the bitstream-3 vector is exact;
- NOT ON SILICON if the fixed-core vector is exact (check the resident bitstream first);
- anything else a FINDING, value by value against the bitstream-3 vector.
The image hash and the resident bitstream are checked by the runner before each boot.
Exit 0 = every image CONFIRMED, 1 = any NOT ON SILICON or FINDING, 2 = any VOID or NO DATA. Written 2026-10-27."""
import importlib.util
import re
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("rg", Path(__file__).with_name("rfix-grade.py"))
rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)
ISSUE = {"fp-of": "R-64", "fp-special": "R-64", "fp-carry": "R-65", "fp-box": "R-66", "fp-edge": "R-65/R-66 (edge)"}


def prereg(path):
    t = open(path).read()
    rows = {m.group(1): int(m.group(2)) for m in
            re.finditer(r"^\| `([a-z0-9-]+)\.bin` \|[^|]*\|[^|]*\| (\d+) \|", t, re.M)}
    vec = {}
    for img, n in rows.items():
        sec = re.search(rf"^### {re.escape(img)}\n(.*?)(?=^### |^## |\Z)", t, re.M | re.S)
        if not sec:
            print(f"FP6466 PREREG: no section for {img}"); sys.exit(2)
        v = {}
        for key, label in (("b3", "Bitstream 3 ("), ("fix", "Fixed core (")):
            m = re.search(re.escape(label) + r"[^\n]*\n\n?`([^`]+)`", sec.group(1))
            if not m:
                print(f"FP6466 PREREG: {img} has no '{label}' vector"); sys.exit(2)
            v[key] = [int(x, 16) for x in m.group(1).split()]
            if len(v[key]) != n:
                print(f"FP6466 PREREG: {img} {label} has {len(v[key])} values, SR {n}"); sys.exit(2)
        vec[img] = (n, v)
    if set(vec) != set(ISSUE):
        print(f"FP6466 PREREG: images {sorted(vec)} are not {sorted(ISSUE)}"); sys.exit(2)
    return vec


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    vec = prereg(sys.argv[1])
    worst = 0
    for img in ISSUE:
        n, v = vec[img]
        r = rg.read(sys.argv[2], img)
        if r is None:
            print(f"FP6466 {img} NO DATA"); worst = 2; continue
        sr, sv = r
        print(f"FP6466 {img} read: SR {sr} :: {' '.join(f'{x:#x}' for x in sv)}")
        if not (len(sr) == 1 and int(sr[0], 16) == n and len(sv) == n):
            print(f"FP6466 {img} VOID: SR {sr} / {len(sv)} values, want {n}"); worst = 2; continue
        if sv == v["b3"]:
            print(f"FP6466 {img} CONFIRMED ON SILICON ({ISSUE[img]}): the bitstream-3 vector exactly")
        elif sv == v["fix"]:
            print(f"FP6466 {img} NOT ON SILICON ({ISSUE[img]}): the fixed-core vector; check the resident bitstream first")
            worst = max(worst, 1)
        else:
            d = [f"{i}: read {a:#x} want {b:#x}" for i, (a, b) in enumerate(zip(sv, v["b3"])) if a != b]
            print(f"FP6466 {img} FINDING: {'; '.join(d)}"); worst = max(worst, 1)
    sys.exit(worst)


if __name__ == "__main__":
    main()
