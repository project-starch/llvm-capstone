#!/usr/bin/env python3
"""Pre-registered checks for T3 on the other 88 programs (PREREG-t1-t2-sampling.md, last section).

  t3all-checks.py <runs.tsv> <twin-checks.tsv> <T1,T3 dir> <T3,T2 dir> <T1,T2 dir> <main per-program.tsv>...

1. correctness: within each program check= is one value across all three arms; where the twin has a hash
   (twin_checks != 0x0), it equals the twin's.
2. instret: per program the three arms' median per-iteration instret agree within 0.1 %.
3. cycles: per program T3/T1 and T2/T3 (point) within 1 % of 1.000; their geomeans over the run in [0.995, 1.005].
4. replicate: this run's T2/T1 Welch interval overlaps the main run's for each program (several main tables may be
   given; the first that has the program is used).
Exit 1 on no data or a program missing an arm. Prediction misses are REPORTED, not errors."""
import csv, sys, math, statistics as st
from collections import defaultdict
runs, twin, d13, d32, d12, *mains = sys.argv[1:]
if not mains: sys.exit(__doc__)
rows = [r for r in csv.DictReader(open(runs), delimiter='\t') if not r['it'].startswith('w')]
if not rows: sys.exit('t3all-checks: no timed rows in ' + runs)
ins, chk, suite = defaultdict(lambda: defaultdict(list)), defaultdict(set), {}
for r in rows:
    ins[r['program']][r['arm']].append(int(r['instret'])); chk[r['program']].add(r['check']); suite[r['program']] = r['suite']
# The twin table is keyed by (suite, bench name): beebs/qsort and rv8/qsort are different programs (so are the two
# miniz). A run's program is the bench name with its suite as prefix for ctl and rv8 (rv8-qsort), and its beebs-libc
# programs are the twin's beebs rows.
tw = {(r['suite'], r['bench_name']): r['twin_checks'] for r in csv.DictReader(open(twin), delimiter='\t')}
def twin_key(p):
    s = 'beebs' if suite[p] == 'beebs-libc' else suite[p]
    return s, p[len(s) + 1:] if p.startswith(s + '-') else p
pp = lambda d: {r['program']: r for r in csv.DictReader(open(f'{d}/per-program.tsv'), delimiter='\t')}
p13, p32, p12 = pp(d13), pp(d32), pp(d12)
main = {}
for f in reversed(mains): main.update({r['program']: r for r in csv.DictReader(open(f), delimiter='\t')})
miss, g13, g32 = [], [], []
print('program\tcheck\ttwin\tins_spread\tT3/T1\tT2/T3\tT2/T1\tlo\thi\tmain_T2/T1\tmain_lo\tmain_hi\toverlap')
for p in sorted(ins):
    if set(ins[p]) != {'T1', 'T2', 'T3'}: sys.exit(f't3all-checks: {p} arms {sorted(ins[p])}')
    m = {a: st.median(ins[p][a]) for a in ('T1', 'T3', 'T2')}
    spread = max(m.values()) / min(m.values()) - 1
    if spread > 0.001: miss.append(f'{p}: instret spread {spread:.4%}')
    c = sorted(chk[p]); t = tw.get(twin_key(p), '?')
    if len(c) != 1: miss.append(f'{p}: check differs across arms {c}')
    elif t not in ('0x0', '?') and c[0] != t: miss.append(f'{p}: check {c[0]} != twin {t}')
    a, b, r12 = float(p13[p]['ratio']), float(p32[p]['ratio']), p12[p]
    g13.append(math.log(a)); g32.append(math.log(b))
    for name, v in (('T3/T1', a), ('T2/T3', b)):
        if abs(v - 1) > 0.01: miss.append(f'{p}: {name} {v:.4f} outside 1 %')
    mm = main.get(p); ov = 'n/a'
    if mm:
        ov = 'yes' if float(r12['lo']) <= float(mm['hi']) and float(mm['lo']) <= float(r12['hi']) else 'NO'
        if ov == 'NO': miss.append(f'{p}: replicate T2/T1 {r12["ratio"]} [{r12["lo"]}, {r12["hi"]}] vs main {mm["ratio"]} [{mm["lo"]}, {mm["hi"]}]')
    else: miss.append(f'{p}: no main-run row')
    print(f'{p}\t{",".join(c)}\t{t}\t{spread:.5f}\t{a:.4f}\t{b:.4f}\t{r12["ratio"]}\t{r12["lo"]}\t{r12["hi"]}\t'
          + (f'{mm["ratio"]}\t{mm["lo"]}\t{mm["hi"]}' if mm else '\t\t') + f'\t{ov}')
G13, G32 = math.exp(st.mean(g13)), math.exp(st.mean(g32))
for name, g in (('T3/T1', G13), ('T2/T3', G32)):
    if not 0.995 <= g <= 1.005: miss.append(f'geomean {name} {g:.4f} outside [0.995, 1.005]')
print(f'\nprograms {len(ins)}; geomean T3/T1 {G13:.4f}, T2/T3 {G32:.4f}')
print('PREDICTION MISSES:' if miss else 'every pre-registered prediction holds')
for x in miss: print('  ' + x)
