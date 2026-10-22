#!/usr/bin/env python3
"""summarize.py: one row per call of every live-peak cell boot (cell-*/), from lp-parse's brackets and the module's stats.
Per call: exit/fault, the largest table instance's peak bracket, the host minted bracket summed over the call's instances, the
module's exact allocations over the call (nodes_allocated_total delta), whether the module count lies in the host bracket, and
the count of MemoryError lines in the call's transcript. NO DATA (exit 2) if a cell has no readings or no instance."""
import json, re, sys, glob, os
D = os.path.dirname(os.path.abspath(__file__)); bad = 0
rows = []
for cell in sorted(glob.glob(f'{D}/cell-*')):
    if not os.path.isdir(cell) or cell.endswith('.out'): continue
    name = os.path.basename(cell)
    lp = f'{cell}.lp.json'
    if not os.path.exists(lp): print(f"NO DATA {name}: no {lp}"); bad = 2; continue
    insts = json.load(open(lp))
    if name.startswith('cell-mruby'):
        rd = glob.glob(f'{cell}/N*/readings.json')
        r = json.load(open(rd[0])) if rd else {}
        top = max(insts, key=lambda d: d['peak_lb'])
        rows.append((name, 'M6 smoke', f"smoke_done {r.get('smoke_done')}", top['peak_lb'], top['peak_ub'],
                     sum(d['minted_lb'] for d in insts), sum(d['minted_ub'] for d in insts), r.get('allocated'), None, len(insts)))
        continue
    try: r = json.load(open(f'{cell}/readings.json'))
    except Exception as e: print(f"NO DATA {name}: {e}"); bad = 2; continue
    serial = open(f'{cell}/guest/serial.log', 'rb').read().decode('utf-8', 'replace').replace('\r', '')
    prev = (r.get('stats0') or {}).get('nodes_allocated_total', 0)
    for i, x in enumerate(r['calls']):
        step = f"c{i:02d}-" + re.sub(r"[^A-Za-z0-9_.+-]", "_", x['name'])[:60]
        ds = [d for d in insts if d['step'] == step]
        s = x.get('stats') or {}; f = x.get('fault') or {}
        mod = s.get('nodes_allocated_total'); delta = None if mod is None else mod - prev; prev = mod if mod is not None else prev
        m = re.search(r'@@STEP-BEGIN ' + re.escape(step) + r'\n(.*?)@@STEP-END', serial, re.S)
        memerr = m.group(1).count('MemoryError') if m else None
        res = f"{x['result'].get('kind')} {x['result'].get('value')}" + (f" fault {f.get('cause')} {f.get('function')}" if f.get('cause') else '')
        if not ds: rows.append((name, x['name'], res, None, None, None, None, delta, memerr, 0)); continue
        top = max(ds, key=lambda d: d['peak_lb'])
        rows.append((name, x['name'], res, top['peak_lb'], top['peak_ub'], sum(d['minted_lb'] for d in ds), sum(d['minted_ub'] for d in ds), delta, memerr, len(ds)))
print("cell\tcall\tresult\tpeak_lb\tpeak_ub\tminted_lb\tminted_ub\tmodule_alloc\tin_bracket\tMemoryError\tinstances")
for row in rows:
    name, call, res, plb, pub, mlb, mub, delta, memerr, n = row
    inb = None if None in (mlb, delta) else (mlb <= delta <= mub)
    if inb is False: bad = max(bad, 1)
    print(f"{name}\t{call[:58]}\t{res}\t{plb}\t{pub}\t{mlb}\t{mub}\t{delta}\t{inb}\t{memerr}\t{n}")
sys.exit(bad)
