#!/usr/bin/env python3
"""check-control.py SERIAL CONTROL_JSON: the live-peak control's verdict against the pre-registered bands (RESULT, 12:08).
Bands are applied to the bracketed readings of lp-parse.py: the WHOLE bracket must lie in the band.
Exit 0 = every check holds, 1 = a check fails, 2 = no data."""
import json, re, sys
text = open(sys.argv[1], 'rb').read().decode('utf-8', 'replace').replace('\r', '')
try: insts = json.load(open(sys.argv[2]))
except Exception as e: print(f"NO DATA: {e}"); sys.exit(2)
by_step = {}
for d in insts: by_step.setdefault(d['step'], []).append(d)
stats, step = {}, None
for line in text.split('\n'):
    m = re.search(r'LP_BEGIN:(\S+)', line)
    if m: step = m.group(1); continue
    if step and step.startswith('stats') and line.startswith('{"version"'):
        stats[step] = json.loads(line)
ok, nodata = True, False
def check(name, cond, detail):
    global ok
    print(f"  {'HOLDS' if cond else 'FAILS'} {name}: {detail}"); ok &= bool(cond)
bands = {'hold-100000': (100000, 110000, 100000, 'HOLD-OK 100000'), 'churn-200000': (0, 2047, 200000, 'CHURN-OK 200000'),
         'hold-150000': (150000, 165000, 150000, 'HOLD-OK 150000')}
prev = stats.get('stats0')
if prev is None: print("NO DATA: no stats0 line"); sys.exit(2)
for s, (lo, hi, mint, marker) in bands.items():
    ds = by_step.get(s, [])
    if not ds: print(f"NO DATA: no table instance in step {s}"); nodata = True; continue
    top = max(ds, key=lambda d: d['peak_lb'])
    check(f"{s} marker", marker in text, marker)
    check(f"{s} peak bracket inside [{lo}, {hi}]", lo <= top['peak_lb'] and top['peak_ub'] <= hi,
          f"peak [{top['peak_lb']}, {top['peak_ub']}] (inst {top['inst']}, {len(ds)} instance(s))")
    mlb, mub = sum(d['minted_lb'] for d in ds), sum(d['minted_ub'] for d in ds)
    st = stats.get(f'stats-{s}')
    if st is None: print(f"NO DATA: no stats after {s}"); nodata = True; continue
    delta = st['nodes_allocated_total'] - prev['nodes_allocated_total']; prev = st
    check(f"{s} module allocations >= {mint}", delta >= mint, f"module delta {delta}")
    check(f"{s} module allocations inside the host minted bracket", mlb <= delta <= mub, f"module {delta}, host [{mlb}, {mub}]")
if nodata: sys.exit(2)
sys.exit(0 if ok else 1)
