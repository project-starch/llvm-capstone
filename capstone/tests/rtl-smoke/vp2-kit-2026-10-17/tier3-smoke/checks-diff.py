#!/usr/bin/env python3
"""checks-diff.py <twin-checks.tsv> <step text (pilot.txt)> ...: per program, silicon's check= against the twin's (the
runtime lane's QEMU functional pass, one value per program). Silicon's value must be ONE per program across every
iteration and both arms (else SPLIT). Prints EQUAL / DIFFERS / NO-HASH (0x0 on both sides: the program prints no state hash, so the comparison tests
nothing) / SPLIT / NO-TWIN per program, then counts. Exit 2 if a step
text has no BENCH line or the twin file has no row (no data is an error, not a zero). Written 2026-10-26."""
import csv
import re
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    twin = {}
    for r in csv.DictReader(open(sys.argv[1]), delimiter="\t"):     # columns suite, bench_name, twin_checks, readings
        v = (r.get("twin_checks") or "").strip()
        if re.fullmatch(r"0x[0-9a-fA-F]+", v):
            twin[r["bench_name"].strip()] = int(v, 16)
    if not twin:
        print(f"NO DATA: no program row in {sys.argv[1]}"); sys.exit(2)
    sil = defaultdict(set)
    for p in sys.argv[2:]:
        t = open(p, errors="replace").read()
        hits = re.findall(r"^BENCH (\S+) it=\S+ .* check=(0x[0-9a-f]+)", t, re.M)
        if not hits:
            print(f"NO DATA: no BENCH line in {p}"); sys.exit(2)
        for name, c in hits:
            sil[name.split("/")[-1]].add(int(c, 16))
    n = defaultdict(int)
    for prog in sorted(sil):
        v = sil[prog]
        if len(v) != 1:
            k = "SPLIT"; msg = " ".join(f"{x:#x}" for x in sorted(v))
        elif prog not in twin:
            k = "NO-TWIN"; msg = f"silicon {next(iter(v)):#x}"
        else:
            s = next(iter(v))
            # 0x0 on both sides is a program that prints no state hash: equal by construction, so it tests nothing.
            k = "NO-HASH" if s == 0 and twin[prog] == 0 else "EQUAL" if s == twin[prog] else "DIFFERS"
            msg = f"silicon {s:#x} twin {twin[prog]:#x}"
        n[k] += 1
        print(f"CHECK {k:8s} {prog:28s} {msg}")
    print("COUNT " + " ".join(f"{k} {n[k]}" for k in ("EQUAL", "DIFFERS", "NO-HASH", "SPLIT", "NO-TWIN")) + f" of {len(sil)}")


if __name__ == "__main__":
    main()
