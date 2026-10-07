#!/usr/bin/env python3
"""The quarantine ratio knob: footprint against the trigger model, and what a
smaller footprint costs in kernel sweep work.

    plot-ratio.py RESULT_DIR OUT_PREFIX [SETTING]

Runs are one program (SETTING, e.g. "./alloc-test 1", selects it when a
directory holds several) under ./traced with revocation enabled.  r comes
from _RUNTIME_QUARANTINE_DENOMINATOR in the argv (numerator 1), from the arm
name (q8, q2, q1), or is the default 1/4.  Only the asynchronous mode (the
CheriBSD default) is drawn; 'sync' and 'off' runs are left out.

Model, with L the live bytes the program holds and q the arena size at the
trigger.  Asynchronous mode flushes the revoked arena only at the next
trigger, so q = rL/(1-2r) and allocated cycles between
  L+q = L(1-r)/(1-2r)   and   L+2q = L/(1-2r).
Measured: min and max of allocated/live_req over the plateau, i.e. samples
whose live bytes are within 5% of their peak after the first completed pass.
Sampling can miss the extremes of short cycles, so the measured band can only
be narrower than the true one.  r >= 1/2 has no steady state; its band is
clipped and annotated.

Top panel: footprint band against the model.  Bottom panel, same x: kernel
sweep pages (ro + rw, from mqstat's MQ-SWEEP-STATS) per 10^6 allocations.
A run without that line, or whose interposition did not fire (calls=0 with
passes>0), is named on stderr and left out of the bottom panel.
"""
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import parse, setting

ARM_R = {'q8': 1 / 8, 'q2': 1 / 2, 'q1': 1.0}
RED = '#c0392b'


def ratio(run):
    m = re.search(r'_RUNTIME_QUARANTINE_DENOMINATOR=(\d+)', run['argv'])
    if m:
        n = re.search(r'_RUNTIME_QUARANTINE_NUMERATOR=(\d+)', run['argv'])
        return (int(n.group(1)) if n else 1) / int(m.group(1))
    return ARM_R.get(run['arm'], 1 / 4)


def sweep(path):
    m = re.search(r'^MQ-SWEEP-STATS (.*)$', path.read_text(errors='replace'), re.M)
    return dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', m.group(1))) if m else None


def model(r):
    return ((1 - r) / (1 - 2 * r), 1 / (1 - 2 * r)) if r < 0.5 else None


def frac(r):
    return f'1/{round(1 / r)}' if r < 1 else '1'


def main():
    src, out = Path(sys.argv[1]), sys.argv[2]
    want = sys.argv[3] if len(sys.argv) > 3 else None
    pts = []
    for p in sorted(src.glob('*.txt')):
        r = parse(p)
        if r['arm'] in ('off', 'sync'):
            continue
        if not (r['ok'] and r['rows']):
            print(f'skipped (incomplete): {p.name}', file=sys.stderr)
            continue
        if want and setting(r['argv']) != want:
            continue
        rows = r['rows']
        peak = max(x['live_req'] for x in rows)
        plateau = [x['allocated'] / x['live_req'] for x in rows
                   if x['live_req'] >= 0.95 * peak and x['dequeue'] >= 2]
        if not plateau:
            print(f'skipped (no plateau samples after the first pass): {p.name}', file=sys.stderr)
            continue
        sw = sweep(p)
        pages = None
        if sw is None:
            print(f'no MQ-SWEEP-STATS: {p.name}', file=sys.stderr)
        elif sw.get('calls', 1) == 0 and sw.get('passes', 0) > 0:
            print(f'interposition did not fire (calls=0): {p.name}', file=sys.stderr)
        else:
            pages = (sw['pages_scan_ro'] + sw['pages_scan_rw']) / (rows[-1]['op'] / 1e6)
        pts.append({'r': ratio(r), 'min': min(plateau), 'max': max(plateau),
                    'mean': sum(plateau) / len(plateau), 'passes': rows[-1]['dequeue'] // 2,
                    'pages': pages, 'name': setting(r['argv'])})
    if not pts:
        sys.exit(f'no usable asynchronous runs in {src}')
    names = {p['name'] for p in pts}
    if len(names) != 1:
        sys.exit(f'several program settings, pass SETTING: {sorted(names)}')
    pts.sort(key=lambda p: p['r'])
    print(f"setting {names.pop()}")
    print(f"{'r':>6} {'min':>6} {'max':>6} {'mean':>6} {'model min':>9} {'model max':>9} {'passes':>6} {'sweep pages/1e6':>15}")
    for p in pts:
        m = model(p['r'])
        ms = f"{m[0]:9.2f} {m[1]:9.2f}" if m else f"{'diverges':>9} {'':9}"
        print(f"{p['r']:6.4f} {p['min']:6.2f} {p['max']:6.2f} {p['mean']:6.2f} {ms} {p['passes']:6d} "
              f"{'-' if p['pages'] is None else round(p['pages']):>15}")

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.2, 6.4), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 2]})
    ymax = 3.0
    rs = [x / 1000 for x in range(40, 496)]
    ax.fill_between(rs, [model(x)[0] for x in rs], [model(x)[1] for x in rs], color=RED, alpha=.12,
                    lw=0, label='model from mrs.c: between L(1−r)/(1−2r) and L/(1−2r)')
    ax.plot(rs, [model(x)[1] for x in rs], color=RED, lw=.8)
    ax.plot(rs, [model(x)[0] for x in rs], color=RED, lw=.8, ls=':')
    steady = [p for p in pts if model(p['r'])]
    ax.vlines([p['r'] for p in steady], [p['min'] for p in steady], [p['max'] for p in steady],
              color='#1d2328', lw=6, alpha=.85, label='measured: min to max over the plateau')
    for p in pts:
        if not model(p['r']):
            ax.annotate(f"no steady state:\nup to {p['max']:.0f}× live", (p['r'], ymax),
                        xytext=(0, -30), textcoords='offset points', ha='center', fontsize=7,
                        arrowprops=dict(arrowstyle='->', lw=.6))
    ax.axhline(4 / 3, color='k', ls='--', lw=.6)
    ax.text(0.51, 4 / 3, 'documented 4/3', ha='right', va='bottom', fontsize=7)
    ax.set_ylim(1, ymax)
    ax.set_ylabel('heap per live byte\n(jemalloc allocated / live)')
    ax.legend(fontsize=7, frameon=False, loc='upper left')

    sp = [p for p in pts if p['pages'] is not None]
    st = [p for p in sp if model(p['r'])]
    bx.plot([p['r'] for p in st], [p['pages'] for p in st], 'o-', color=RED, ms=5)
    for p in sp:
        if not model(p['r']):
            bx.plot(p['r'], p['pages'], 'o', mfc='white', mec=RED, ms=5)
        bx.annotate(f"{p['pages'] / 1000:.0f}k", (p['r'], p['pages']), xytext=(6, 4),
                    textcoords='offset points', fontsize=7)
    bx.set_ylim(0, max(p['pages'] for p in sp) * 1.2 if sp else 1)
    bx.set_ylabel('kernel sweep work\n(pages scanned per 10⁶ allocations)')
    ticks = sorted({p['r'] for p in pts})
    bx.set_xticks(ticks, [frac(r) + ('\n(default)' if abs(r - 0.25) < 1e-9 else '') for r in ticks])
    bx.set_xlim(min(ticks) * 0.6, max(ticks) * 1.08)
    bx.set_xlabel('quarantine ratio r')
    fig.tight_layout()
    fig.savefig(out + '-ratio.png', dpi=150)


if __name__ == '__main__':
    main()
