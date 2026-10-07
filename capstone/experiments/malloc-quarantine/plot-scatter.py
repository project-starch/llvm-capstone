#!/usr/bin/env python3
"""Peak heap against peak live bytes, one point per program, log-log.

    plot-scatter.py OUT_PNG RESULT_DIR...

Input: traced runs of the default configuration.  x: peak live bytes the
program held (exact peak_live from MQ-DONE when present, else the sampled
peak).  y: peak of jemalloc 'allocated' (live + quarantine).  Reference
lines: y = x (no overhead), y = 2x (model peak of the default, steady live
set), and the 8 MiB floor below which MRS never revokes.  Programs that
appear in several runs (e.g. mstress at several SCALEs) are drawn as one
series.  Runs without MQ-DONE are listed on stderr and left out.
"""
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import parse, setting

MIB = 2**20
FLOOR = 8


def prog(argv):
    return setting(argv).split()[0].lstrip('./')


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    pts = {}
    for d in dirs:
        for p in sorted(Path(d).glob('*.txt')):
            r = parse(p)
            if r['arm'] == 'off':
                continue
            if not (r['ok'] and r['rows']):
                print(f'skipped (incomplete): {p.name}', file=sys.stderr)
                continue
            live = (r['peak_live'] or max(x['live_req'] for x in r['rows'])) / MIB
            alloc = max(x['allocated'] for x in r['rows']) / MIB
            if not r['peak_live'] and len(r['rows']) < 20 and live < alloc / 2:
                print(f'skipped ({len(r["rows"])} samples, no exact peak_live, sampled live far below '
                      f'allocated): {p.name}', file=sys.stderr)
                continue
            if live <= 0:
                print(f'skipped (no live bytes seen): {p.name}', file=sys.stderr)
                continue
            pts.setdefault(prog(r['argv']), []).append((live, alloc, setting(r['argv'])))
    if not pts:
        sys.exit('no usable runs')
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    lo, hi = 0.05, 2000
    xs = [lo * (hi / lo) ** (i / 200) for i in range(201)]
    ax.plot(xs, xs, color='#888888', lw=0.9, ls='-', label='no overhead (heap = live)')
    ax.plot(xs, [2 * x for x in xs], color='#c0392b', lw=0.9, ls='--', label='model peak of the default: 2 × live')
    ax.axhline(FLOOR, color='#2471a3', lw=0.9, ls=':', label='8 MiB: MRS never revokes below this')
    print(f"{'program':14} {'peak live MiB':>13} {'peak alloc MiB':>14} {'ratio':>6}")
    for name, series in sorted(pts.items(), key=lambda kv: min(s[0] for s in kv[1])):
        series.sort()
        for live, alloc, argv in series:
            print(f"{name:14} {live:13.2f} {alloc:14.2f} {alloc / live:6.2f}   {argv}")
        marker = 's' if len(series) > 1 else 'o'
        ax.plot([s[0] for s in series], [s[1] for s in series], marker, color='#1d2328', ms=5,
                ls='-' if len(series) > 1 else 'none', lw=0.8)
        live, alloc, _ = series[-1]
        ax.annotate(name, (live, alloc), xytext=(5, -3), textcoords='offset points', fontsize=8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(1, hi)
    ax.set_xlabel('peak bytes the program holds (MiB)')
    ax.set_ylabel('peak heap jemalloc holds (MiB): live + quarantine')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.grid(True, which='major', lw=0.3, color='#dddddd')
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == '__main__':
    main()
