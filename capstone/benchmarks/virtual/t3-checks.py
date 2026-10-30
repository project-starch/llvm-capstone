#!/usr/bin/env python3
"""T3 pre-registered checks on a three-arm run TSV (parse-board rows, warm-ups included).

  t3-checks.py <runs.tsv> <main per-program.tsv> <T1,T3 dir> <T3,T2 dir> <T1,T2 dir>

1. instret: per program, median per-iteration instret per arm; prediction T1 <= T3 <= T2.
2. miniz-libc: share of log(T2/T1) carried by T2/T3; prediction > 0.80, i.e. T3/T1 < 1.17; entries per arm.
3. replicate: this run's T2/T1 against the main run's, inside the union of the two Welch intervals.
Exit 1 on no data or a program missing an arm. Prediction misses are REPORTED, not errors."""
import csv, sys, math, statistics as st
from collections import defaultdict
runs, mainpp, d13, d32, d12 = sys.argv[1:6]
rows = [r for r in csv.DictReader(open(runs), delimiter='\t') if not r['it'].startswith('w')]
if not rows: sys.exit('t3-checks: no timed rows in ' + runs)
ins, ent = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
for r in rows:
    ins[r['program']][r['arm']].append(int(r['instret'])); ent[r['program']][r['arm']].append(int(r['entries']))
def pp(d): return {r['program']: r for r in csv.DictReader(open(f'{d}/per-program.tsv'), delimiter='\t')}
p13, p32, p12, pm = pp(d13), pp(d32), pp(d12), {r['program']: r for r in csv.DictReader(open(mainpp), delimiter='\t')}
print('program\tins_T1\tins_T3\tins_T2\torder_ok\tT3/T1\tT2/T3\tT2/T1\tmain_T2/T1\treplicate_ok\tent_T1\tent_T3\tent_T2')
miss = []
for p in sorted(ins):
    if set(ins[p]) != {'T1', 'T2', 'T3'}: sys.exit(f't3-checks: {p} arms {sorted(ins[p])}')
    m = {a: st.median(ins[p][a]) for a in ('T1', 'T3', 'T2')}
    e = {a: st.median(ent[p][a]) for a in ('T1', 'T3', 'T2')}
    ok = m['T3'] >= m['T1'] * 0.999 and m['T2'] >= m['T3'] * 0.999   # PREREG: 0.1 % tolerance
    if not ok: miss.append(f'{p}: instret order T1 {m["T1"]:.0f} T3 {m["T3"]:.0f} T2 {m["T2"]:.0f}')
    r12, mm = p12[p], pm.get(p)
    rep = 'n/a'
    if mm:
        lo, hi = min(float(r12['lo']), float(mm['lo'])), max(float(r12['hi']), float(mm['hi']))
        rep = 'yes' if lo <= float(r12['ratio']) <= hi and lo <= float(mm['ratio']) <= hi and \
              max(float(r12['lo']), float(mm['lo'])) <= min(float(r12['hi']), float(mm['hi'])) else 'NO (intervals disjoint)'
        if rep != 'yes': miss.append(f'{p}: replicate T2/T1 {r12["ratio"]} [{r12["lo"]}, {r12["hi"]}] vs main {mm["ratio"]} [{mm["lo"]}, {mm["hi"]}]')
    print(f'{p}\t{m["T1"]:.0f}\t{m["T3"]:.0f}\t{m["T2"]:.0f}\t{"yes" if ok else "NO"}\t{p13[p]["ratio"]}\t{p32[p]["ratio"]}\t'
          f'{r12["ratio"]}\t{mm["ratio"] if mm else "-"}\t{rep}\t{e["T1"]:.0f}\t{e["T3"]:.0f}\t{e["T2"]:.0f}')
if 'miniz-libc' in p12:
    t31, t21 = float(p13['miniz-libc']['ratio']), float(p12['miniz-libc']['ratio'])
    share = 1 - math.log(t31) / math.log(t21)
    print(f'\nminiz-libc: T3/T1 {t31:.4f} [{p13["miniz-libc"]["lo"]}, {p13["miniz-libc"]["hi"]}], T2/T1 {t21:.4f}; '
          f'scrub share of log T2/T1 = {share:.3f} (prediction > 0.80, T3/T1 < 1.17): {"HOLDS" if share > 0.8 else "FAILS"}')
print('\nprediction misses:' + ('\n  ' + '\n  '.join(miss) if miss else ' none'))
