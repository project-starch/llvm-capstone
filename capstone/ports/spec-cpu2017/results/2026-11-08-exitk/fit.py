import json, sys, numpy as np
c = json.load(open(sys.argv[1]))
skip_first = '--all' not in sys.argv
rows = c[1:] if skip_first else c
F = 25e6
def fit(yname, y, cols, names):
    X = np.array(cols, dtype=float).T
    coef, res, rk, sv = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print('%-14s R2=%.4f  ' % (yname, r2) + '  '.join('%s=%.4g' % (n, v) for n, v in zip(names, coef)))
    return coef
cyc = np.array([x['cyc'] for x in rows], float)
steps = np.array([x['steps'] for x in rows], float)
ent = np.array([x['entries'] for x in rows], float)
ev1 = np.array([x['ev'][1] for x in rows], float)
ev3 = np.array([x['ev'][3] for x in rows], float)
ev4 = np.array([x['ev'][4] for x in rows], float)
k = np.array([x['kcyc'] - x['rcyc'] for x in rows], float)
r = np.array([x['rcyc'] for x in rows], float)
u = np.array([x['ucyc'] for x in rows], float)
o = np.array([x['ocyc'] for x in rows], float)
one = np.ones_like(cyc)
print('contexts', len(rows), 'sum inside %.1f s, k-r %.1f s, rcyc %.1f s, ucyc %.1f s, ocyc %.1f s, steps %d entries %d ev1 %d ev3 %d ev4 %d' % (
    cyc.sum()/F, k.sum()/F, r.sum()/F, u.sum()/F, o.sum()/F, steps.sum(), ent.sum(), ev1.sum(), ev3.sum(), ev4.sum()))
ticks = cyc / 250000.0  # masked 10 ms periods at 25 MHz
fit('kcyc-rcyc', k, [one, steps, ent, ticks], ['const', 'per_step', 'per_entry', 'per_masked_tick'])
fit('kcyc-rcyc', k, [one, steps, ticks], ['const', 'per_step', 'per_masked_tick'])
fit('ucyc', u, [one, steps, ticks], ['const', 'per_step', 'per_masked_tick'])
fit('ucyc', u, [one, ev3, steps, ticks], ['const', 'per_svc', 'per_step', 'per_masked_tick'])
fit('k+u-r', k + u, [one, steps, ticks], ['const', 'per_step', 'per_masked_tick'])
fit('rcyc', r, [one, ev4], ['const', 'per_fault'])
print('-- with quantum exits (ev1) as a regressor')
fit('kcyc-rcyc', k, [one, steps, ev1, ticks], ['const', 'per_step', 'per_qexit', 'per_masked_tick'])
fit('kcyc-rcyc', k, [one, steps, ev1, ev3, ev4, ticks], ['const', 'per_step', 'per_qexit', 'per_svc', 'per_fault', 'per_masked_tick'])
fit('ucyc', u, [one, steps, ev1, ev3, ticks], ['const', 'per_step', 'per_qexit', 'per_svc', 'per_masked_tick'])
