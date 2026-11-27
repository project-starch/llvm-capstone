#!/usr/bin/env python3
"""tick-bins.py REPLAY_TICKS [QEMU_TICKS] (nested lane, 2026-11-27): where the 1024x8 verdict cache's removed misses lie
between timer ticks, against how memory-bound each inter-tick interval is.

REPLAY_TICKS is rcm-lb-dc-ticks --ticks output: one row per MTI edge, before it is applied, with cumulative counts
  tick event_index lastI lastP checks LB0_misses LB_misses removed_lb dmiss_rec0..3
QEMU_TICKS is the tracer's <trace>.ticks side file (CAPSTONE_RECTRACE_TICKS=1): one row per MTI with
  put event_index instructions_all instructions_protected dc_reads dc_miss0..3 prog_reads prog_miss0
The two are joined on the event index of the MTI edge. Every replay row must find its QEMU row, or the tool exits 2.

An interval is the stretch from one tick to the next. Per interval: instructions (exact from QEMU_TICKS, else the last
I mark, placed every 1024 runs), removed_lb, LB0 misses, and (with QEMU_TICKS) the D$ model's misses, all sources, seed
0. Intervals are sorted by a memory-boundness proxy and cut into deciles of INSTRUCTIONS, so each decile is a tenth of
the program's instructions; each decile prints its share of removed_lb and of LB0 misses, and its proxy's mean.
Without QEMU_TICKS the proxy is instructions per interval (short = slow emulation, the v3 wall-clock reading) and
the record-line D$ miss rate. No data is an error: an empty or unjoinable input exits 2.
"""
import sys


def rows(path, n):
    out = []
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) != n:
                sys.exit(f'{path}: a row with {len(p)} fields, not {n}: {ln!r}')
            out.append([int(x) for x in p])
    if not out:
        sys.exit(f'{path}: no rows')
    return out


def deciles(iv, key, label):
    """iv: list of dicts with ins, rem, lb0, and key fields; sorted by key, cut by instruction share."""
    iv = sorted(iv, key=lambda d: d[key])
    T = sum(d['ins'] for d in iv); R = sum(d['rem'] for d in iv); L = sum(d['lb0'] for d in iv)
    if T <= 0 or R <= 0:
        sys.exit('no instructions or no removed misses in the intervals')
    print(f'deciles by {label} (low to high); each is 10% of instructions')
    dc = 'dcm' in iv[0]
    print(f'{"dec":>3} {"intervals":>9} {"ins_share":>9} {"removed_share":>13} {"LB0miss_share":>13} {"removed/kins":>12} {"key_mean":>12}'
          + (f' {"dc_mpki":>9}' if dc else ''))
    acc = 0; b = 0; cur = dict(n=0, ins=0, rem=0, lb0=0, k=0.0, dcm=0)
    out = []
    for d in iv:
        cur['n'] += 1; cur['ins'] += d['ins']; cur['rem'] += d['rem']; cur['lb0'] += d['lb0']; cur['k'] += d[key] * d['ins']
        cur['dcm'] += d.get('dcm', 0)
        acc += d['ins']
        if acc >= T * (b + 1) / 10 and b < 9:
            out.append(cur); cur = dict(n=0, ins=0, rem=0, lb0=0, k=0.0, dcm=0); b += 1
    out.append(cur)
    for i, c in enumerate(out):
        if not c['ins']:
            continue
        print(f'{i:>3} {c["n"]:>9} {c["ins"] / T:>9.4f} {c["rem"] / R:>13.4f} {c["lb0"] / L:>13.4f} '
              f'{1000 * c["rem"] / c["ins"]:>12.4f} {c["k"] / c["ins"]:>12.4f}'
              + (f' {1000 * c["dcm"] / c["ins"]:>9.3f}' if dc else ''))
    # rank correlation of key and removed density, interval-weighted, as one number
    return out


def complement(iv):
    """The slice is the top decile of instructions by LB0 misses per 1000 instructions, the rest is everything else.
    A CPI proxy per side is 1 + (LB0 misses * lbc + D$ misses * dcc) / instructions, for listed costs (not measured),
    with D$ misses either the program source's (prog, seed 0) or all sources' (all, seed 0). BUILD on the rtl lane's
    uniform curve needs the slice's tick density <= 411/660 of the mean; with density proportional to CPI that is
    X / (0.1 X + 0.9 Y) <= 0.6227, i.e. Y / X >= (1 / 0.6227 - 0.1) / 0.9."""
    iv = sorted(iv, key=lambda d: d['lb0_pki'], reverse=True)
    T = sum(d['ins'] for d in iv); acc = 0; k = 0
    while acc < T / 10:
        acc += iv[k]['ins']; k += 1
    sides = {'slice': iv[:k], 'rest': iv[k:]}
    agg = {n: {f: sum(d[f] for d in v) for f in ('ins', 'lb0', 'rem', 'dcm', 'pm')} for n, v in sides.items()}
    for n, a in agg.items():
        print(f'complement {n}: intervals {len(sides[n])} ins {a["ins"]} LB0 {a["lb0"]} removed {a["rem"]} '
              f'LB0/kins {1000 * a["lb0"] / a["ins"]:.3f} prog_dmiss/kins {1000 * a["pm"] / a["ins"]:.3f} '
              f'all_dmiss/kins {1000 * a["dcm"] / a["ins"]:.3f}')
    need = (1 / (411 / 660) - 0.1) / 0.9
    print(f'complement BUILD needs Y/X >= {need:.4f}')
    print(f'{"lb_cost":>7} {"dc_cost":>7} {"src":>4} {"X_slice":>8} {"Y_rest":>8} {"Y/X":>7} {"lean":>8}')
    for lbc in (40.4, 15.6):
        for dcc in (20, 40, 80):
            for src, f in (('prog', 'pm'), ('all', 'dcm')):
                X = 1 + (agg['slice']['lb0'] * lbc + agg['slice'][f] * dcc) / agg['slice']['ins']
                Y = 1 + (agg['rest']['lb0'] * lbc + agg['rest'][f] * dcc) / agg['rest']['ins']
                print(f'{lbc:>7} {dcc:>7} {src:>4} {X:>8.3f} {Y:>8.3f} {Y / X:>7.3f} {"BUILD" if Y / X >= need else "against":>8}')
    # calibrated rows: the D$ cost at which the whole program's proxy equals a mean CPI of 5.7 (165.03 G silicon cycles
    # over about 29 G instructions, the rtl lane's estimate; the silicon instruction count is not measured)
    I = agg['slice']['ins'] + agg['rest']['ins']
    for lbc in (40.4, 15.6):
        for src, f in (('prog', 'pm'), ('all', 'dcm')):
            L = agg['slice']['lb0'] + agg['rest']['lb0']; M = agg['slice'][f] + agg['rest'][f]
            dcc = (4.7 * I - L * lbc) / M if M else float('nan')
            X = 1 + (agg['slice']['lb0'] * lbc + agg['slice'][f] * dcc) / agg['slice']['ins']
            Y = 1 + (agg['rest']['lb0'] * lbc + agg['rest'][f] * dcc) / agg['rest']['ins']
            print(f'{lbc:>7} {dcc:>7.2f} {src:>4} {X:>8.3f} {Y:>8.3f} {Y / X:>7.3f} {"BUILD" if Y / X >= need else "against":>8} calibrated to mean 5.7')


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    rp = rows(sys.argv[1], 12)
    q = None
    if len(sys.argv) == 3:
        qr = rows(sys.argv[2], 11)
        q = {r[1]: r for r in qr if r[0] == 1}
        miss = [r for r in rp if r[1] not in q]
        print(f'join: replay rows {len(rp)} qemu rows {len(qr)} (put {len(q)}, skipped {len(qr) - len(q)}) unjoined {len(miss)}')
        if miss:
            sys.exit(f'{len(miss)} replay rows have no QEMU row, first {miss[0]}')
    iv = []
    for a, b in zip(rp, rp[1:]):
        d = dict(rem=b[7] - a[7], lb0=b[5] - a[5], chk=b[4] - a[4], dm=b[8] - a[8])
        if q:
            qa, qb = q[a[1]], q[b[1]]
            d['ins'] = qb[2] - qa[2]
            d['dcm'] = qb[5] - qa[5]
            d['dcr'] = qb[4] - qa[4]
            d['pm'] = qb[10] - qa[10]
        else:
            d['ins'] = b[2] - a[2]
        if d['ins'] <= 0:
            continue                       # two ticks inside one I-mark gap (mark-placed counts only)
        d['ipt'] = d['ins']
        d['rec_dmiss_rate'] = d['dm'] / d['chk'] if d['chk'] else 0.0
        d['lb0_pki'] = 1000 * d['lb0'] / d['ins']
        if q:
            d['dc_mpki'] = 1000 * d['dcm'] / d['ins']
        iv.append(d)
    T = sum(d['ins'] for d in iv); R = sum(d['rem'] for d in iv)
    print(f'intervals {len(iv)} instructions {T} removed_lb {R} (in intervals; the replay total includes before the first '
          f'and after the last tick)')
    if q:
        deciles(iv, 'dc_mpki', 'D$ misses per 1000 instructions (all sources, seed 0)')
    deciles(iv, 'lb0_pki', 'LB0 misses per 1000 instructions')
    if q:
        complement(iv)
    deciles(iv, 'ipt', 'instructions per interval')
    print('END')


main()
