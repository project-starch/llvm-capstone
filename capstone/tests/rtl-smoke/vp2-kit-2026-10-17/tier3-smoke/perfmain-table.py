#!/usr/bin/env python3
"""perfmain-table.py <step text (pilot.txt)> <steps.tsv> [program ...]: per program and arm, each process's outcome from a
perf kit's step text: rc, BENCH lines (want W+I from steps.tsv), VERIFY-FAILED count, and per timed iteration the
entries and instret (min-max). One line per process, then a per-program summary: processes passing / failing per arm.
With program names, only those (the runtime lane's FP list). Exit 2 if the step text has no @@PERF-BEGIN (no data is an
error, not a zero). Written 2026-10-26."""
import csv
import re
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    text = open(sys.argv[1], errors="replace").read()
    plan = {int(r["n"]): r for r in csv.DictReader(open(sys.argv[2]), delimiter="\t")}
    only = set(sys.argv[3:])
    blocks = re.split(r"(?m)^@@PERF-BEGIN ", text)[1:]
    if not blocks:
        print(f"NO DATA: no @@PERF-BEGIN in {sys.argv[1]}"); sys.exit(2)
    summ = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for blk in blocks:
        n = int(blk.split(None, 1)[0])
        r = plan[n]
        prog, arm = r["program"], r["arm"]
        if only and prog not in only:
            continue
        want = sum(int(x) for x in r["args"].split())
        end = re.search(r"^@@PERF-END \d+ .* rc=(\d+)", blk, re.M)
        its = [(m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)) for m in re.finditer(
            r"^BENCH \S+ it=(\S+) cycles=\d+ instret=(\d+) entries=(\d+) ns=\d+ check=0x[0-9a-f]+(.*)$", blk, re.M)]
        timed = [i for i in its if not i[0].startswith("w")]
        vf = len(re.findall(r"VERIFY-FAILED", blk))     # on a BENCH line (the real format) or anywhere in the process's output
        fault = re.search(r"domain fault cause=(\d+)", blk)
        rc = end.group(1) if end else "none"
        ok = rc == "0" and len(its) == want and vf == 0
        summ[prog][arm][0 if ok else 1] += 1
        ent = f"{min(i[2] for i in timed)}-{max(i[2] for i in timed)}" if timed else "-"
        ins = f"{min(i[1] for i in timed)}-{max(i[1] for i in timed)}" if timed else "-"
        print(f"PROC {n:3d} {prog:22s} {arm} p{r['proc']} K{r['K']}: {'PASS' if ok else 'FAIL'} rc={rc} BENCH {len(its)}/{want} "
              f"VERIFY-FAILED {vf}{' fault cause ' + fault.group(1) if fault else ''} entries/it {ent} instret/it {ins}")
    for prog in sorted(summ):
        a = summ[prog]
        print(f"PROGRAM {prog:22s} T1 pass/fail {a['T1'][0]}/{a['T1'][1]}  T2 pass/fail {a['T2'][0]}/{a['T2'][1]}")


if __name__ == "__main__":
    main()
