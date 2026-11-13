#!/usr/bin/env python3
"""symdiff.py NATIVE.capmix NMAP BASE.capmix BMAP [SCALE]: kernel instructions by symbol (from System.map) in mark window 0 of two
per-PC (shift 2) capmix3 files, side by side; base also scaled by SCALE (pages ratio); sorted by native - SCALE*base."""
import sys, bisect, os
NC = 14
def load(path, smap):
    snaps, cur = [], None
    for line in open(path):
        w = line.split()
        if not w: continue
        if w[0] in ('SNAP', 'FINAL'): cur = {'tag': w[0], 'label': int(w[2]), 'b': {}}
        elif w[0] == 'B': cur['b'][int(w[1], 16)] = sum(map(int, w[2:2 + NC]))
        elif w[0] == 'END': snaps.append(cur)
    s = [x for x in snaps if x['tag'] == 'SNAP']
    a, b = s[0], s[1]
    assert (a['label'], b['label']) == (1, 2)
    syms = sorted((int(x.split()[0], 16), x.split()[2]) for x in open(smap) if x.split()[1] in 'tTwW')
    addrs = [x[0] for x in syms]
    ker = {}
    for k, v in b['b'].items():
        n = v - a['b'].get(k, 0)
        pc = (k << 2) % (1 << 64)
        if not n: continue
        if pc >= 0xffff000000000000 or 0x80200000 <= pc < 0x81000000:
            va = pc if pc >= 0xffff000000000000 else pc - 0x80200000 + 0xffffffff80000000
            i = bisect.bisect_right(addrs, va) - 1
            ker[syms[i][1] if i >= 0 else '?'] = ker.get(syms[i][1] if i >= 0 else '?', 0) + n
    return ker
n, b = load(sys.argv[1], sys.argv[2]), load(sys.argv[3], sys.argv[4])
sc = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
print(f'native {sum(n.values()):,} base {sum(b.values()):,} base*{sc} {sum(b.values())*sc:,.0f}')
keys = sorted(set(n) | set(b), key=lambda k: -(n.get(k, 0) - sc * b.get(k, 0)))
K = int(os.environ.get('TOPK', '40'))
for k in keys[:K] + ['...'] + keys[-15:]:
    if k == '...': print('  ...'); continue
    print(f'  {n.get(k,0):>13,} {b.get(k,0):>13,} {n.get(k,0)-sc*b.get(k,0):>+14,.0f}  {k}')
