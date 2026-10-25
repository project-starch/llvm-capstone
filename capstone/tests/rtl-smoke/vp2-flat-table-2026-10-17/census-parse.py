#!/usr/bin/env python3
"""census-parse.py ROOT [tags]: the stats.collections census over ROOT/t3-<tag>/**/serial.log VP2_NODES lines.
Per arm: steps begun, VP2_NODES lines (baseline + one per step), peak per-step nodes allocated, peak high water,
collections and reclaimed deltas. A step without its VP2_NODES line, or an arm with no serial, is NO-DATA (exit 2)."""
import json, re, sys
from pathlib import Path
root = Path(sys.argv[1]); tags = sys.argv[2:] or sorted(p.name[3:] for p in root.glob('t3-*') if p.is_dir())
nodata = nonzero = 0
for t in tags:
    logs = sorted((root / ('t3-' + t)).rglob('serial.log'))
    if not logs: print(f'{t:6} NO-DATA (no serial.log)'); nodata += 1; continue
    tot = {'steps': 0, 'lines': 0, 'missing': 0, 'alloc_max': 0, 'hw': 0, 'coll': 0, 'recl': 0, 'cap': set()}
    for s in logs:
        text = s.read_text(errors='replace')
        begun = re.findall(r'^@@STEP-BEGIN (\S+)$', text, re.M)
        rows = {m[1]: json.loads(m[2]) for m in re.finditer(r'^VP2_NODES:([^:]+):(\{.*\})$', text, re.M)}
        tot['steps'] += len(begun); tot['lines'] += len(rows)
        missing = [b for b in begun if b not in rows] + ([] if 'baseline' in rows else ['baseline'])
        tot['missing'] += len(missing)
        prev = rows.get('baseline')
        for b in begun:
            d = rows.get(b)
            if d is None or prev is None: prev = d or prev; continue
            tot['alloc_max'] = max(tot['alloc_max'], d['nodes_allocated_total'] - prev['nodes_allocated_total'])
            tot['coll'] += d['collections'] - prev['collections']; tot['recl'] += d['nodes_reclaimed'] - prev['nodes_reclaimed']
            tot['hw'] = max(tot['hw'], d['nodes_high_water']); tot['cap'].add(d['node_capacity']); prev = d
    flag = 'NO-DATA' if tot['missing'] or not tot['lines'] else ('COLLECTED' if tot['coll'] else 'zero')
    nodata += flag == 'NO-DATA'; nonzero += flag == 'COLLECTED'
    print(f"{t:6} {flag:9} steps {tot['steps']:3} lines {tot['lines']:3} missing {tot['missing']} "
          f"peak-step-nodes {tot['alloc_max']:5} high-water {tot['hw']:5} collections {tot['coll']} reclaimed {tot['recl']} cap {sorted(tot['cap'])}")
print(f'TOTAL arms {len(tags)} collected {nonzero} no-data {nodata}')
sys.exit(2 if nodata else 0)
