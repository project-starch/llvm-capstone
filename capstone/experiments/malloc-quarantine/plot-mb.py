#!/usr/bin/env python3
"""Normalised peak RSS of the mimalloc-bench programs, sorted by baseline size.

    plot-mb.py RESULT_DIR OUT_PREFIX

One group of bars per program (on, sync), each the kernel's max RSS divided
by the same program's 'off' run.  The baseline MiB is printed under each
program; the dashed line is the documented 4/3 target of a 1/4 quarantine.
Programs without a complete 'off' run are skipped and named on stderr.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader

amb = SourceFileLoader('amb', str(Path(__file__).with_name('analyze-mb.py'))).load_module()
COLORS = {'on': '#c0392b', 'sync': '#2471a3'}
LABELS = {'on': 'default (async, 1/4)', 'sync': 'sync, 1/4'}


def main():
    src, out = Path(sys.argv[1]), sys.argv[2]
    runs = [amb.parse(p) for p in sorted(src.glob('*.txt'))]
    ok = [r for r in runs if r['ok']]
    base = {r['argv']: r['rss'] for r in ok if r['arm'] == 'off'}
    progs = sorted(base, key=base.get)
    for r in runs:
        if r['argv'] not in base:
            print(f"skipped (no complete off run): {r['argv']} {r['arm']}", file=sys.stderr)
    if not progs:
        sys.exit(f'no complete off runs in {src}')
    fig, ax = plt.subplots(figsize=(1.1 * len(progs) + 2, 3.6))
    w = 0.38
    for i, arm in enumerate(('on', 'sync')):
        xs, ys = [], []
        for j, p in enumerate(progs):
            r = next((r for r in ok if r['argv'] == p and r['arm'] == arm), None)
            if r:
                xs.append(j + (i - 0.5) * w); ys.append(r['rss'] / base[p])
                print(f"{p:40} {arm:5} {r['rss'] / base[p]:6.2f}")
        bars = ax.bar(xs, ys, w, color=COLORS[arm], label=LABELS[arm])
        ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=1)
    ax.axhline(4 / 3, ls='--', lw=0.8, color='k')
    ax.axhline(1, lw=0.6, color='#888888')
    ax.set_xticks(range(len(progs)))
    ax.set_xticklabels([f"{p.split()[0].lstrip('./')}\n{base[p] / 1024:.1f} MiB" for p in progs],
                       fontsize=8)
    ax.set_ylabel('max RSS / revocation off')
    ax.set_yscale('log', base=2)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out + '-rss.png', dpi=150)


if __name__ == '__main__':
    main()
