#!/usr/bin/env python3
"""Cross the domain arms with the native verdicts.

level0 is the control: free only marks, nothing is revoked, so a fault there
is not a catch -- it is what the defect does unprotected. A catch is a case
that completes in level0 and faults in a revoking arm.
"""
import json, re, sys, os
K='/tmp/capstone/mruby-arms'; W='/tmp/capstone/mruby-corpus'

def arm(path):
    out={}
    if not os.path.exists(path): return out
    for line in open(path):
        m=re.match(r'CASE (\S+) status=(\S+) fault=(\S+) first=(.*)$', line.strip())
        if m:
            out[m.group(1)]=dict(status=int(m.group(2)), fault=m.group(3),
                                 first=m.group(4))
    return out

arms={a:arm(f'{K}/results/{a}.txt') for a in ('level0','sublet','sublet-gc')}
native={r['name']:r for r in json.load(open(f'{W}/verdicts-pin-host.json'))}
mat={r['name']:r for r in json.load(open(f'{W}/matrix.json'))}

def verdict(e):
    if e is None: return 'missing'
    if e['fault']!='none': return 'FAULT('+e['fault']+')'
    if e['status']==137: return 'WATCHDOG'
    if e['status']>=128: return f"SIGNAL({e['status']-128})"
    if e['status']!=0: return f"exit({e['status']})"
    return 'PASS' if e['first']=='none' or '"PASS"' in e['first'] else 'FAIL'

rows=[]
for n in sorted(arms['level0']):
    r=dict(name=n, subject=native[n]['subject'],
           asan=mat[n]['asan'], asan_page1=mat[n]['asan_page1'],
           native=native[n]['verdict'])
    for a in arms: r[a]=verdict(arms[a].get(n))
    rows.append(r)
json.dump(rows, open(f'{K}/arms-matrix.json','w'), indent=1)

print(f"{len(rows)} cases run in the domain\n")
for a in arms:
    if not arms[a]: continue
    from collections import Counter
    print(f"  {a:10}", dict(Counter(r[a] for r in rows)))
caught=[r for r in rows if r['level0'] in ('PASS','FAIL')
        and (r.get('sublet','missing').startswith('FAULT')
             or r.get('sublet-gc','missing').startswith('FAULT'))]
print(f"\ncaught by a revoking arm and not by the control: {len(caught)}")
for r in caught:
    print(f"  {r['name']} level0={r['level0']:8} sublet={r.get('sublet','-'):12} "
          f"sublet-gc={r.get('sublet-gc','-'):12} asan={r['asan'] or '-':20} {r['subject'][:52]}")
