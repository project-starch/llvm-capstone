#!/usr/bin/env python3
# judge-m1f.py <extract.txt>... : per arm, the snapshot count, minted-revoked on every snapshot, and take/give cycles
# per operation (take_cyc/n, give_cyc/n) as median [min, max]. Exit 2 when a file has no R1 m1 line.
import re, sys, statistics as st
from collections import defaultdict
for f in sys.argv[1:]:
    t = open(f).read()
    snaps = [dict(kv.split('=', 1) for kv in l.split()[3:] if '=' in kv) for l in t.splitlines() if l.startswith('R1 m1 snap')]
    ends = [l for l in t.splitlines() if not l.startswith('R1 m1 snap') and (l.startswith('R1 m1 ') or 'probe' in l or l.startswith('#') or 'cause' in l or 'refus' in l.lower())]
    if not snaps and not ends: print(f"{f}: NO DATA"); sys.exit(2)
    by = defaultdict(list)
    for s in snaps: by[s['arm']].append(s)
    print(f"== {f.split('/')[-1]}: {len(snaps)} snapshots")
    for arm, ss in by.items():
        mr = sorted(set(int(s['minted']) - int(s['revoked']) for s in ss))
        tk = [int(s['take_cyc']) / int(s['n']) for s in ss if int(s.get('n', 0))]
        gv = [int(s['give_cyc']) / int(s['n']) for s in ss if int(s.get('n', 0))]
        fmt = lambda v: f"{st.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]" if v else "-"
        print(f"  {arm}: snapshots {len(ss)}  minted-revoked {mr}  take/op {fmt(tk)}  give/op {fmt(gv)}")
    for l in ends: print("  " + l[:200])
