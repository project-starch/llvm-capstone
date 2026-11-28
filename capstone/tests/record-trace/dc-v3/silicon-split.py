#!/usr/bin/env python3
"""silicon-split.py (nested lane, 2026-11-27): silicon's own CPI on mcf's LB-miss slice and on the rest, from the
runtime lane's tick log (b5b2-ticklog-2026-11-27, read-ticklog.py), with the slice taken from the twin's mcf-tk6.

  silicon-split.py --twin REPLAY_TICKS QEMU_TICKS (--uart CAPTURE --reader READ_TICKLOG_PY | --selftest)

The twin side: the joined mcf-tk6 intervals (tick-bins.py's join), each with exact instructions_all and
instructions_protected, LB0 misses, removed_lb and all-source D$ misses. The slice is tick-bins.py's: the top decile of
instructions by LB0 misses per 1000 instructions.

The silicon side: the window's per-tick records (mcycle, minstret, hpm3 deltas; read-ticklog.py's table(), markers()
and window(), cut at the mcf stage's own up0/up1 exactly as that reader cuts it).

Alignment: by position in the window's instructions, NOT by tick rank (silicon's ticks are every 250,000 cycles, the
twin's every 62,500 instructions). Silicon tick k covers [G_k, G_k+1) of its window's cumulative minstret,
normalised to [0, 1]; the twin's intervals cover [F_j, F_j+1) of their cumulative count, normalised the same way.
Each silicon tick takes the slice weight w_k = the share of its range that overlaps twin slice intervals. Two keys
for the twin's count: instructions_all (as silicon's minstret, all modes, but the twin's kernel ticks at 250 Hz and
silicon's at 100 Hz) and instructions_protected (the program only). Both are reported; the difference is the
alignment's sensitivity to the kernel share.

Readings per key: X and Y by least squares on cycles_k = X * w_k * instret_k + Y * (1 - w_k) * instret_k (a tick that
mixes slice and rest code biases the plain weighted ratios toward the mean; those are printed beside); Y / X against 1.6732 (BUILD needs
Y / X >= 1.6732, the rtl lane's uniform-curve threshold, tick-bins.py's complement form); silicon's hpm3 per 1000
instructions on each side.
Alignment check: the Pearson correlation, over silicon ticks, of silicon's hpm3 per instruction with the twin's
all-source D$ misses per instruction mapped onto the same range. A NEGATIVE CONTROL shifts the twin's map by half the
window (circularly): its correlation must fall and its Y/X move toward 1. If the aligned correlation is not clearly
above the shifted one, the split is reported UNALIGNED and not read.
--selftest builds a silicon log from the twin intervals with known CPIs (slice 2.0, rest 6.0, ticks every 250,000
cycles) and must recover them; the shifted control must not.
"""
import argparse
import bisect
import types
import sys

NEED = (1 / (411 / 660) - 0.1) / 0.9


def rows(path, n):
    out = []
    for ln in open(path):
        p = ln.split()
        if len(p) != n:
            sys.exit(f'{path}: a row with {len(p)} fields, not {n}')
        out.append([int(x) for x in p])
    if not out:
        sys.exit(f'{path}: no rows')
    return out


def twin_intervals(rp_path, qt_path):
    rp = rows(rp_path, 12)
    q = {r[1]: r for r in rows(qt_path, 11) if r[0] == 1}
    miss = [r for r in rp if r[1] not in q]
    if miss:
        sys.exit(f'{len(miss)} replay rows have no QEMU row')
    iv = []
    for a, b in zip(rp, rp[1:]):
        qa, qb = q[a[1]], q[b[1]]
        ia, ip = qb[2] - qa[2], qb[3] - qa[3]
        if ia <= 0:
            continue
        iv.append(dict(ia=ia, ip=ip, lb0=b[5] - a[5], rem=b[7] - a[7], dcm=qb[5] - qa[5]))
    srt = sorted(range(len(iv)), key=lambda j: 1000 * iv[j]['lb0'] / iv[j]['ia'], reverse=True)
    T = sum(d['ia'] for d in iv); acc = 0
    for j in srt:
        if acc >= T / 10:
            break
        acc += iv[j]['ia']; iv[j]['slice'] = 1
    for d in iv:
        d.setdefault('slice', 0)
    return iv


def cum(xs):
    out, c = [0.0], 0
    tot = float(sum(xs))
    for x in xs:
        c += x; out.append(c / tot)
    return out


def overlap_weights(G, F, vals):
    """for each silicon range [G_k, G_k+1), the measure-weighted mean of vals over the twin ranges [F_j, F_j+1)"""
    out = []
    for k in range(len(G) - 1):
        lo, hi = G[k], G[k + 1]
        if hi <= lo:
            out.append(0.0); continue
        j = max(bisect.bisect_right(F, lo) - 1, 0)
        s = 0.0
        while j < len(F) - 1 and F[j] < hi:
            a, b = max(lo, F[j]), min(hi, F[j + 1])
            if b > a:
                s += (b - a) * vals[j]
            j += 1
        out.append(s / (hi - lo))
    return out


def pearson(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = sum((a - mx) ** 2 for a in x) ** 0.5; sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sxy / (sx * sy) if sx and sy else float('nan')


def split(sil, iv, key, shift=0.0):
    """sil: list of (cycles, instret, hpm3). Returns the readings for one twin key, optionally shifted."""
    G = cum([r[1] for r in sil])
    F = cum([d[key] for d in iv])
    iv2 = iv
    if shift:                                   # the negative control: the twin's intervals rotated by half the window
        h = int(len(iv) * shift)
        iv2 = iv[h:] + iv[:h]
        F = cum([d[key] for d in iv2])
    w = overlap_weights(G, F, [d['slice'] for d in iv2])
    tdm = overlap_weights(G, F, [d['dcm'] / d[key] if d[key] else 0.0 for d in iv2])
    cyc = [r[0] for r in sil]; ins = [r[1] for r in sil]; hpm = [r[2] for r in sil]
    sw = lambda v, ww: sum(a * b for a, b in zip(v, ww))
    W1 = w; W0 = [1 - x for x in w]
    Xr = sw(cyc, W1) / sw(ins, W1); Yr = sw(cyc, W0) / sw(ins, W0)    # weighted ratios: biased toward the mean by mixed ticks
    # least squares for c_k = X * w_k i_k + Y * (1 - w_k) i_k (unbiased under that model)
    a = [x * i for x, i in zip(W1, ins)]; b = [x * i for x, i in zip(W0, ins)]
    saa, sbb, sab = sw(a, a), sw(b, b), sw(a, b); sac, sbc = sw(a, cyc), sw(b, cyc)
    det = saa * sbb - sab * sab
    X = (sac * sbb - sbc * sab) / det; Y = (sbc * saa - sac * sab) / det
    hx = 1000 * sw(hpm, W1) / sw(ins, W1); hy = 1000 * sw(hpm, W0) / sw(ins, W0)
    r = pearson([h / i if i else 0.0 for h, i in zip(hpm, ins)], tdm)
    return dict(X=X, Y=Y, YX=Y / X, Xr=Xr, Yr=Yr, hx=hx, hy=hy, r=r, slice_ins=sw(ins, W1) / sum(ins))


def report(variants, iv):
    """variants: (label, sil, twin key, graded). Graded: X / mean <= 411/660 (exact for any slice share), with Y/X
    against 1.6732 printed beside (equivalent when the slice holds 10 % of the instructions)."""
    ok = True
    print(f'BUILD needs X / mean <= {411 / 660:.4f} (graded), i.e. Y/X >= {NEED:.4f} at a 10 % slice')
    for label, sil, key, graded in variants:
        a = split(sil, iv, key); s = split(sil, iv, key, shift=0.5)
        mean = sum(r[0] for r in sil) / sum(r[1] for r in sil)
        aligned = a['r'] > s['r'] + 0.2 and a['r'] > 0.3
        if graded:
            ok &= aligned
        print(f'{label}: ticks {len(sil)} instructions {sum(r[1] for r in sil):,.0f} cycles {sum(r[0] for r in sil):,.0f} '
              f'mean CPI {mean:.4f}{"" if graded else " (reported, not graded)"}')
        for tag, d in (('aligned', a), ('shifted control', s)):
            print(f'  {tag}: X_slice {d["X"]:.4f} Y_rest {d["Y"]:.4f} X/mean {d["X"] / mean:.4f} Y/X {d["YX"]:.4f} '
                  f'({"BUILD" if d["X"] / mean <= 411 / 660 else "against"}) (weighted ratios {d["Xr"]:.4f} / {d["Yr"]:.4f}) '
                  f'hpm3/kins slice {d["hx"]:.3f} rest {d["hy"]:.3f} slice share {d["slice_ins"]:.4f} r {d["r"]:.4f}')
        print(f'  {"ALIGNED" if aligned else "UNALIGNED (not read)"}')
    return ok


def selftest(iv):
    """silicon CPI 2.0 on slice instructions, 6.0 elsewhere; ticks every 250,000 cycles; hpm3 = twin D$ misses."""
    stream = []  # (instructions, cpi, dmiss per instruction) in twin order
    for d in iv:
        stream.append((d['ia'], 2.0 if d['slice'] else 6.0, d['dcm'] / d['ia']))
    sil = []; c = 0.0; i = 0.0; h = 0.0
    for n, cpi, m in stream:
        left = n
        while left > 0:
            room = (250_000 - c) / cpi
            take = min(left, room)
            c += take * cpi; i += take; h += take * m; left -= take
            if c >= 250_000 - 1e-6:
                sil.append((round(c), round(i), round(h))); c = i = h = 0.0
    if i:
        sil.append((round(c), round(i), round(h)))
    a = split(sil, iv, 'ia'); s = split(sil, iv, 'ia', shift=0.5)
    print(f'selftest aligned X {a["X"]:.4f} Y {a["Y"]:.4f} (ratios {a["Xr"]:.4f} / {a["Yr"]:.4f}) r {a["r"]:.4f}; '
          f'shifted X {s["X"]:.4f} Y {s["Y"]:.4f} r {s["r"]:.4f}')
    ok = abs(a['X'] - 2.0) < 0.05 and abs(a['Y'] - 6.0) < 0.05 and s['r'] < a['r'] - 0.2 and abs(s['YX'] - 1) < abs(a['YX'] - 1)
    print('selftest', 'PASS' if ok else 'FAIL')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--twin', nargs=2, required=True)
    ap.add_argument('--uart'); ap.add_argument('--reader'); ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--fw-fixed', type=float); ap.add_argument('--k-entry', type=float)
    ap.add_argument('--icount-shift', type=int, default=0)
    a = ap.parse_args()
    iv = twin_intervals(*a.twin)
    if a.selftest:
        sys.exit(0 if selftest(iv) else 1)
    if not (a.uart and a.reader and a.fw_fixed is not None and a.k_entry is not None):
        sys.exit('--uart, --reader, --fw-fixed and --k-entry (as read-ticklog.py takes them), or --selftest')
    # read-ticklog.py calls main() at module level; load its text without that final call (its file is not touched)
    src = open(a.reader).read()
    if not src.rstrip().endswith('\nmain()'):
        sys.exit(f'{a.reader}: does not end in a bare main() call; refusing to guess how to import it')
    m = types.ModuleType('rtl_reader'); exec(compile(src.rstrip()[:-len('main()')], a.reader, 'exec'), m.__dict__)
    log = open(a.uart, 'rb').read().decode(errors='replace').replace('\r', '')
    hd, recs, sha = m.table(log)
    ctl, rc, up0, up1 = m.markers(log)
    f = (hd['cf'] - hd['c0']) / ((hd['tf'] - hd['t0']) / 1e9)
    times, c = [hd['t0'] / 1e9], 0
    for dc, *_ in recs:
        c += dc; times.append(hd['t0'] / 1e9 + c / f)
    W = m.window(times, up0, up1)
    if len(W) < 2:
        sys.exit(f'{len(W)} interrupts in the window')
    wr = recs[W[0]:W[-1]]
    U = 1 << a.icount_shift
    prog = lambda r: (r[1] - r[4] - r[8] - r[10]) / U - a.fw_fixed * r[7] - a.k_entry * r[6]   # read-ticklog.py's prog()
    variants = [
        ('program key, total cycles (GRADED)', [(r[0] / U, prog(r), r[2]) for r in wr], 'ip', True),
        ('program key, program cycles', [((r[0] - r[3]) / U, prog(r), r[2]) for r in wr], 'ip', False),
        ('all-mode key, total cycles', [(r[0] / U, r[1] / U, r[2]) for r in wr], 'ia', False),
    ]
    print(f'table sha256 {sha} window [{up0}, {up1}] rc={rc}; program sum {sum(prog(r) for r in wr):,.0f} '
          f'(twin key 26,644,263,390); twin intervals {len(iv)} slice {sum(d["slice"] for d in iv)}')
    sys.exit(0 if report(variants, iv) else 1)


main()
