#!/usr/bin/env python3
"""dcstar.py BINS_OUT (nested lane, 2026-11-27): from tick-bins.py's 'complement slice' and 'complement rest' lines, the
D$ miss cost dc* at which the proxy's Y/X crosses the BUILD threshold, for each LB cost and D$ source.

Proxy (tick-bins.py's): CPI = 1 + (LB0/kins * lb + dmiss/kins * dc) / 1000. With a = 1 + LB0/kins * lb / 1000 and
b = dmiss/kins / 1000 on each side, Y/X = need  <=>  dc* = (need * a_s - a_r) / (b_r - need * b_s).
- If b_r <= need * b_s, Y/X tends to b_r / b_s < need as dc grows, so it never reaches the threshold at any dc >= 0
  unless it already does at dc = 0: 'never' (against BUILD at any D$ cost, under this no-overlap proxy).
- If dc* < 0, Y/X is above the threshold at every dc >= 0: 'always'.
Report-only: it grades nothing. A missing line is an error (exit 2).
"""
import re
import sys

NEED = (1 / (411 / 660) - 0.1) / 0.9
side = {}
for ln in open(sys.argv[1]):
    m = re.match(r'complement (slice|rest): .* LB0/kins ([\d.]+) prog_dmiss/kins ([\d.]+) all_dmiss/kins ([\d.]+)', ln)
    if m:
        side[m.group(1)] = dict(lb0=float(m.group(2)), prog=float(m.group(3)), all=float(m.group(4)))
if set(side) != {'slice', 'rest'}:
    sys.exit(f'{sys.argv[1]}: complement slice/rest lines not found')
print(f'dcstar need Y/X >= {NEED:.4f}')
for lb in (40.4, 15.6):
    for src in ('prog', 'all'):
        s, r = side['slice'], side['rest']
        a_s, a_r = 1 + s['lb0'] * lb / 1000, 1 + r['lb0'] * lb / 1000
        b_s, b_r = s[src] / 1000, r[src] / 1000
        den = b_r - NEED * b_s
        num = NEED * a_s - a_r
        if den <= 0:
            verdict = 'always' if num <= 0 else f'never (Y/X -> {b_r / b_s if b_s else float("inf"):.4f} < {NEED:.4f})'
        else:
            dc = num / den
            verdict = 'always' if dc < 0 else f'dc* {dc:.2f} cycles'
        ratio = b_r / b_s if b_s else float('inf')
        print(f'dcstar lb_cost {lb} src {src}: {verdict}; MPKI ratio rest/slice {ratio:.4f}')
print('END')
