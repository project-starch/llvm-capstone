#!/usr/bin/env python3
"""silicon-split-diag.py (nested lane, 2026-11-27): POST-HOC diagnostics of why silicon-split.py's registered alignment
gate read UNALIGNED on the program key (r 0.2215) while the all-mode key read ALIGNED (r 0.4640). It grades nothing and
does not change the registered reading, which stays "not read". Same arguments as silicon-split.py's real-capture mode.

Prints, for the program key, the gate correlation recomputed in variants:
  - as registered (silicon all-mode hpm3 per program instruction against the twin's all-source D$ misses per
    protected instruction);
  - Spearman rank correlation of the same pair (robust to outliers);
  - the same Pearson without the ticks that carry a non-timer kernel entry (K-entries > 1: page faults, ecalls);
  - silicon PROGRAM hpm3 (hpm3 - Khpm3) per program instruction against the twin's PROGRAM-source D$ misses per
    protected instruction (the units matched on both sides);
and the split's X, Y and X/mean beside each, so the reader can see whether the reading depends on the gate variant."""
import sys
import types

src = open(sys.argv[sys.argv.index('--split') + 1]).read()
sys.argv = [a for i, a in enumerate(sys.argv) if a != '--split' and (i == 0 or sys.argv[i - 1] != '--split')]
assert src.rstrip().endswith('\nmain()')
S = types.ModuleType('split'); exec(compile(src.rstrip()[:-len('main()')], 'silicon-split.py', 'exec'), S.__dict__)
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--twin', nargs=2, required=True); ap.add_argument('--uart', required=True); ap.add_argument('--reader', required=True)
ap.add_argument('--fw-fixed', type=float, required=True); ap.add_argument('--k-entry', type=float, required=True)
a = ap.parse_args()

# the twin side, with the program-source misses kept (silicon-split.py's twin_intervals drops them)
rp = S.rows(a.twin[0], 12); q = {r[1]: r for r in S.rows(a.twin[1], 11) if r[0] == 1}
iv = S.twin_intervals(*a.twin)
k = 0
for x, y in zip(rp, rp[1:]):
    qa, qb = q[x[1]], q[y[1]]
    if qb[2] - qa[2] <= 0:
        continue
    iv[k]['pm'] = qb[10] - qa[10]; k += 1
assert k == len(iv)

rsrc = open(a.reader).read(); assert rsrc.rstrip().endswith('\nmain()')
m = types.ModuleType('rd'); exec(compile(rsrc.rstrip()[:-len('main()')], a.reader, 'exec'), m.__dict__)
log = open(a.uart, 'rb').read().decode(errors='replace').replace('\r', '')
hd, recs, sha = m.table(log); ctl, rc, up0, up1 = m.markers(log)
f = (hd['cf'] - hd['c0']) / ((hd['tf'] - hd['t0']) / 1e9)
times, c = [hd['t0'] / 1e9], 0
for dc, *_ in recs:
    c += dc; times.append(hd['t0'] / 1e9 + c / f)
W = m.window(times, up0, up1); wr = recs[W[0]:W[-1]]
prog = lambda r: (r[1] - r[4] - r[8] - r[10]) - a.fw_fixed * r[7] - a.k_entry * r[6]


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); rk = [0.0] * len(v)
    for j, i in enumerate(o):
        rk[i] = j
    return rk


def gate(sil, key, twin_field, mask=None):
    G = S.cum([r[1] for r in sil]); F = S.cum([d[key] for d in iv])
    tdm = S.overlap_weights(G, F, [d[twin_field] / d[key] if d[key] else 0.0 for d in iv])
    hs = [r[2] / r[1] if r[1] > 0 else 0.0 for r in sil]
    idx = [i for i in range(len(sil)) if mask is None or mask[i]]
    x = [hs[i] for i in idx]; y = [tdm[i] for i in idx]
    return S.pearson(x, y), S.pearson(ranks(x), ranks(y)), len(idx)


sil_reg = [(r[0], prog(r), r[2]) for r in wr]
sil_prog_h = [(r[0], prog(r), r[2] - r[5]) for r in wr]
nontimer = [r[6] <= 1 for r in wr]
print(f'table {sha[:16]} window ticks {len(wr)}; ticks with a non-timer kernel entry {sum(1 for x in nontimer if not x)}')
for name, sil, tf, mask in (('registered pair', sil_reg, 'dcm', None),
                            ('registered pair, non-timer ticks dropped', sil_reg, 'dcm', nontimer),
                            ('program hpm3 vs twin program misses', sil_prog_h, 'pm', None),
                            ('program hpm3 vs twin program misses, non-timer ticks dropped', sil_prog_h, 'pm', nontimer)):
    p, s, n = gate(sil, 'ip', tf, mask)
    rot = iv[len(iv) // 2:] + iv[:len(iv) // 2]
    ivs, iv_ = iv, rot
    globals()['iv'] = rot; pr, sr, _ = gate(sil, 'ip', tf, mask); globals()['iv'] = ivs
    print(f'{name}: ticks {n} Pearson {p:.4f} (rotated {pr:.4f}) Spearman {s:.4f} (rotated {sr:.4f})')
d = S.split(sil_reg, iv, 'ip'); mean = sum(r[0] for r in sil_reg) / sum(r[1] for r in sil_reg)
print(f'split (graded variant, as registered): X {d["X"]:.4f} Y {d["Y"]:.4f} X/mean {d["X"] / mean:.4f}')
print('END (post hoc; the registered reading is unchanged)')
