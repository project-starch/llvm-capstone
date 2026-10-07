#!/usr/bin/env python3
"""Timeline and footprint factor for traced runs (needs matplotlib).

    plot.py RESULT_DIR OUT_PREFIX

timeline: jemalloc 'allocated' (live + MRS quarantine) and the live request
          against allocation count, one panel per program setting, one line
          per arm.
factor:   kernel max RSS of each arm divided by the 'off' arm, against the
          peak live set (only drawn when 'off' runs exist).
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import parse, setting

COLORS = {'off': '#555555', 'on': '#c0392b', 'sync': '#2471a3', 'q8': '#d68910',
          'q2': '#7d3c98', 'q1': '#148f77'}
MIB = 2**20


def main():
    src, out = Path(sys.argv[1]), sys.argv[2]
    runs = [r for r in (parse(p) for p in sorted(src.glob('*.txt'))) if r['ok'] and r['rows']]
    if not runs:
        sys.exit(f'no complete runs in {src}')
    for r in runs:
        r['argv'] = setting(r['argv'])
    peak = {}
    for r in runs:
        peak[r['argv']] = max(peak.get(r['argv'], 0), max(x['live_req'] for x in r['rows']))
    argvs = sorted(peak, key=peak.get)

    fig, axes = plt.subplots(1, len(argvs), figsize=(4.2 * len(argvs), 3.4), squeeze=False)
    for ax, argv in zip(axes[0], argvs):
        for r in sorted((r for r in runs if r['argv'] == argv), key=lambda r: r['arm']):
            ops = [x['op'] for x in r['rows']]
            ax.plot(ops, [x['allocated'] / MIB for x in r['rows']], color=COLORS.get(r['arm']),
                    label=f"allocated ({r['arm']})", lw=1.3)
        r0 = next(r for r in runs if r['argv'] == argv)
        ax.plot([x['op'] for x in r0['rows']], [x['live_req'] / MIB for x in r0['rows']],
                color='black', ls=':', lw=1, label='live bytes held')
        ax.axhline(8, color='grey', lw=0.8, ls='--', label='MRS 8 MiB floor')
        ax.set_title(f"{argv} (peak live {peak[argv] / MIB:.1f} MiB)", fontsize=9)
        ax.set_xlabel('allocations')
        ax.set_ylim(bottom=0)
    axes[0][0].set_ylabel('jemalloc allocated (MiB)')
    h, l = axes[0][-1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc='upper center', ncol=len(l))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out + '-timeline.png', dpi=150)

    base = {r['argv']: r['maxrss_kib'] for r in runs if r['arm'] == 'off'}
    if not base:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for arm in sorted({r['arm'] for r in runs} - {'off'}):
        pts = sorted((peak[r['argv']] / MIB, r['maxrss_kib'] / base[r['argv']])
                     for r in runs if r['arm'] == arm and r['argv'] in base)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', color=COLORS.get(arm), label=arm)
    ax.axhline(4 / 3, color='grey', ls='--', lw=0.8)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('peak live set of the off run (MiB)')
    ax.set_ylabel('max RSS / revocation off')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out + '-factor.png', dpi=150)


if __name__ == '__main__':
    main()
