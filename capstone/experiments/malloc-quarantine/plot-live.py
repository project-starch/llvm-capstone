#!/usr/bin/env python3
"""Footprint per live byte for traced runs, with revocation enabled in every arm.

    plot-live.py RESULT_DIR OUT_PREFIX

For each program and arm: the kernel's max RSS and jemalloc's peak 'allocated'
(live + quarantine), both divided by the peak live bytes the program held
(mqtrace's live_req, usable sizes).  No 'off' run is needed; the factor is the
whole cost of the allocator plus quarantine per byte the program keeps.
Programs are sorted by peak live bytes; the dashed line is the model peak of
the default (async, 1/4) for a steady live set: 2 (4/3 for sync, if present).  Runs without MQ-DONE are
reported as FAILED on stderr, never drawn as zero.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import parse, setting

ARMS = [('on', 'default (async, 1/4)', '#c0392b'), ('sync', 'sync, 1/4', '#2471a3'),
        ('q8', 'async, 1/8', '#d68910')]
MIB = 2**20


def main():
    src, out = Path(sys.argv[1]), sys.argv[2]
    runs = {}
    for p in sorted(src.glob('*.txt')):
        r = parse(p)
        if r['arm'] == 'off':
            continue
        if not (r['ok'] and r['rows']):
            print(f'FAILED {p.name}', file=sys.stderr)
            continue
        live = r['peak_live'] or max(x['live_req'] for x in r['rows'])
        runs[(setting(r['argv']), r['arm'])] = {
            'live': live, 'rss': r['maxrss_kib'] * 1024 / live,
            'ledger': max(x['allocated'] for x in r['rows']) / live,
            'passes': r['rows'][-1]['dequeue'] // 2}
    if not runs:
        sys.exit(f'no complete traced runs in {src}')
    progs = sorted({k[0] for k in runs}, key=lambda p: max(v['live'] for k, v in runs.items() if k[0] == p))
    print(f"{'program':44} {'arm':5} {'peak live MiB':>13} {'maxRSS/live':>11} {'alloc/live':>10} {'passes':>6}")
    for p in progs:
        for arm, _, _ in ARMS:
            v = runs.get((p, arm))
            if v:
                print(f"{p[:44]:44} {arm:5} {v['live'] / MIB:13.2f} {v['rss']:11.2f} {v['ledger']:10.2f} {v['passes']:6d}")
    fig, axes = plt.subplots(2, 1, figsize=(1.3 * len(progs) + 2, 6.4), sharex=True)
    arms = [a for a in ARMS if any(k[1] == a[0] for k in runs)]
    w = 0.8 / len(arms)
    for ax, key, label in ((axes[0], 'rss', 'max RSS / peak live'),
                           (axes[1], 'ledger', 'jemalloc allocated / peak live')):
        for i, (arm, name, color) in enumerate(arms):
            xs, ys = [], []
            for j, p in enumerate(progs):
                v = runs.get((p, arm))
                if v:
                    xs.append(j + (i - (len(arms) - 1) / 2) * w); ys.append(v[key])
            bars = ax.bar(xs, ys, w, color=color, label=name)
            ax.bar_label(bars, fmt='%.2f', fontsize=6, padding=1)
        if any(k[1] == 'sync' for k in runs):
            ax.axhline(4 / 3, ls='--', lw=.7, color='#2471a3')
        ax.axhline(2, ls='--', lw=.7, color='#c0392b', label='model peak, steady live set')
        ax.axhline(1, lw=.6, color='#888888')
        ax.set_ylim(0, None)
        ax.set_ylabel(label)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc='upper center', ncol=len(l))
    axes[1].set_xticks(range(len(progs)))
    axes[1].set_xticklabels([f"{p.split()[0].lstrip('./')}\n{max(v['live'] for k, v in runs.items() if k[0] == p) / MIB:.1f} MiB live"
                             for p in progs], fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out + '-live.png', dpi=150)


if __name__ == '__main__':
    main()
