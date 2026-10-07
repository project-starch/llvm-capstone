#!/usr/bin/env python3
"""Spatial locality of consecutive allocations: how far apart are the
addresses that malloc hands out one after another?

    plot-stride.py RESULT_DIR OUT_PNG

Input: MQ_TRACK=1 runs under ./traced2 (MQ-STRIDE: log2 buckets of
|address - previous allocation's address|; bucket 0 = same address, bucket b
= [2^(b-1), 2^b)).  One CDF per program, 'on' runs only.  Prints the median
bucket and the share of consecutive allocations within 64 B, 4 KiB and 2 MiB.
"""
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import parse, setting

COLORS = ['#c0392b', '#2471a3', '#d68910', '#148f77', '#7d3c98', '#1d2328']


def hist(path):
    m = re.search(r'^MQ-STRIDE(.*)$', path.read_text(errors='replace'), re.M)
    return {int(b): int(c) for b, c in re.findall(r'(\d+):(\d+)', m.group(1))} if m else None


def main():
    src, out = Path(sys.argv[1]), sys.argv[2]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    i = 0
    for p in sorted(src.glob('*.txt')):
        r = parse(p)
        h = hist(p)
        if not (r['ok'] and r['rows'] and h) or r['arm'] == 'off':
            if r['arm'] != 'off':
                print(f'FAILED or no MQ-STRIDE: {p.name}', file=sys.stderr)
            continue
        total = sum(h.values())
        xs, ys, acc = [], [], 0
        for b in range(66):
            acc += h.get(b, 0)
            xs.append(2 ** b); ys.append(acc / total)
        med = next(x for x, y in zip(xs, ys) if y >= 0.5)
        within = {n: sum(c for b, c in h.items() if 2 ** b <= n) / total for n in (64, 4096, 2 ** 21)}
        name = setting(r['argv']).split()[0].lstrip('./')
        print(f"{name:14} allocations={total} median_stride<={med} within64B={within[64]:.3f} "
              f"within4KiB={within[4096]:.3f} within2MiB={within[2 ** 21]:.3f}")
        ax.step(xs, ys, where='post', color=COLORS[i % len(COLORS)], lw=1.6, label=name)
        i += 1
    for v, txt in ((64, '64 B line'), (4096, '4 KiB page'), (2 ** 21, '2 MiB')):
        ax.axvline(v, color='#999999', lw=.7, ls=':')
        ax.text(v * 1.15, 0.04, txt, fontsize=7, color='#666666')
    ax.set_xscale('log', base=2)
    ax.set_xlim(1, 2 ** 32)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('distance between consecutive allocations (bytes)')
    ax.set_ylabel('share of allocations')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == '__main__':
    main()
