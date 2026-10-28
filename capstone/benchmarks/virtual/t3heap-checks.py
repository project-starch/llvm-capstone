#!/usr/bin/env python3
"""Pre-registered checks for the 15 allocating programs on bitstream 3 (PREREG-t1-t2-sampling.md, last section).

  t3heap-checks.py <runs.tsv> <ref runs.tsv> <run dir> <ref dir> <all-103 dir>

<runs.tsv> is this run's rows and <ref runs.tsv> the bitstream-2 T3 run's (parse-board rows, warm-ups included).
<run dir> and <ref dir> each hold T1-T3/, T3-T2/, T1-T2/ from perf-analyze `report --arms`; <all-103 dir> holds the
same three over this run's rows plus the T3-all run's.

1. correctness: within each program check= is one value across the three arms, equal to the reference run's.
2. instret: per program and arm, the median per-iteration instret equals the reference run's within 0.1 %.
3. replicate: per program and pair, this run's Welch interval overlaps the reference run's.
4. suite geomeans over all 103 programs within 0.005 of the pre-registered table (coremark is not a prediction).
Exit 1 on no data, a program missing an arm, or a program absent from the reference. Misses are REPORTED."""
import csv, sys, statistics as st
from collections import defaultdict
if len(sys.argv) != 6: sys.exit(__doc__)
runs, ref, rdir, fdir, adir = sys.argv[1:]
PAIRS = ('T1-T3', 'T3-T2', 'T1-T2')
PRED = {'beebs': (1.0001, 0.9999, 1.0000), 'beebs-libc': (1.0265, 1.0775, 1.1061), 'rv8': (0.9997, 1.0247, 1.0244),
        'ctl': (0.9980, 1.0156, 1.0136)}
def load(f):
    ins, chk = defaultdict(lambda: defaultdict(list)), defaultdict(set)
    for r in csv.DictReader(open(f), delimiter='\t'):
        if r['it'].startswith('w'): continue
        ins[r['program']][r['arm']].append(int(r['instret'])); chk[r['program']].add(r['check'])
    if not ins: sys.exit(f't3heap-checks: no timed rows in {f}')
    return ins, chk
ins, chk = load(runs); rins, rchk = load(ref)
pp = lambda d, pr: {r['program']: r for r in csv.DictReader(open(f'{d}/{pr}/per-program.tsv'), delimiter='\t')}
cur = {pr: pp(rdir, pr) for pr in PAIRS}; old = {pr: pp(fdir, pr) for pr in PAIRS}
miss = []
print('program\tcheck\tref_check\tins_dev_max\t' + '\t'.join(f'{pr}\t{pr}_ref\toverlap' for pr in PAIRS))
for p in sorted(ins):
    if set(ins[p]) != {'T1', 'T2', 'T3'}: sys.exit(f't3heap-checks: {p} arms {sorted(ins[p])}')
    if p not in rins: sys.exit(f't3heap-checks: {p} not in the reference run')
    c, rc = sorted(chk[p]), sorted(rchk[p])
    if len(c) != 1: miss.append(f'{p}: check differs across arms {c}')
    elif c != rc: miss.append(f'{p}: check {c} != reference {rc}')
    dev = max(abs(st.median(ins[p][a]) / st.median(rins[p][a]) - 1) for a in ('T1', 'T3', 'T2'))
    if dev > 0.001: miss.append(f'{p}: instret differs from the reference by {dev:.4%}')
    cols = []
    for pr in PAIRS:
        a, b = cur[pr][p], old[pr][p]
        ov = float(a['lo']) <= float(b['hi']) and float(b['lo']) <= float(a['hi'])
        if not ov: miss.append(f'{p}: {pr} {a["ratio"]} [{a["lo"]}, {a["hi"]}] vs bitstream 2 {b["ratio"]} [{b["lo"]}, {b["hi"]}]')
        cols.append(f'{a["ratio"]}\t{b["ratio"]}\t{"yes" if ov else "NO"}')
    print(f'{p}\t{",".join(c)}\t{",".join(rc)}\t{dev:.5f}\t' + '\t'.join(cols))
print('\nsuite geomeans over all 103 programs, bitstream 3 (predicted in parentheses)')
seen = set()
for i, pr in enumerate(PAIRS):
    for r in csv.DictReader(open(f'{adir}/{pr}/per-suite.tsv'), delimiter='\t'):
        g = float(r['geomean']); seen.add(r['suite'])
        if r['suite'] in PRED:
            want = PRED[r['suite']][i]
            if abs(g - want) > 0.005: miss.append(f'suite {r["suite"]} {pr} {g:.4f}, predicted {want:.4f} +- 0.005')
            print(f'  {r["suite"]}\t{pr}\tn={r["programs"]}\t{g:.4f} [{r["lo"]}, {r["hi"]}]\t({want:.4f})')
        else:
            print(f'  {r["suite"]}\t{pr}\tn={r["programs"]}\t{g:.4f} [{r["lo"]}, {r["hi"]}]\t(not predicted)')
if not set(PRED) <= seen: sys.exit(f't3heap-checks: suites missing from {adir}: {sorted(set(PRED) - seen)}')
print('\nPREDICTION MISSES:' if miss else '\nevery pre-registered prediction holds')
for x in miss: print('  ' + x)
