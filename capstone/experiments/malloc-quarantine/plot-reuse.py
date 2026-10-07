#!/usr/bin/env python3
"""Reuse distance and per-window working set for MQ_TRACK=1 traced runs.

    plot-reuse.py RESULT_DIR OUT_PREFIX

Left: CDF of the distance (in allocations, log2 buckets) between the free of
an address and its next allocation.  Right: distinct 64 B lines handed out
per sampling window, relative to the 'off' arm of the same program.
Also prints the numbers the figure is drawn from.
"""
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import parse, setting

COLORS = {'off': '#555555', 'on': '#c0392b', 'sync': '#2471a3'}


def hist(path):
    m = re.search(r'^MQ-HIST(.*)$', path.read_text(), re.M)
    return {int(b): int(c) for b, c in re.findall(r'(\d+):(\d+)', m.group(1))} if m else None


def main():
    src, out = Path(sys.argv[1]), sys.argv[2]
    runs = []
    for p in sorted(src.glob('*.txt')):
        r = parse(p)
        if r['ok'] and r['rows']:
            r['hist'] = hist(p)
            r['argv'] = setting(r['argv'])
            runs.append(r)
    if not runs:
        sys.exit(f'no complete runs in {src}')
    argvs = sorted({r['argv'] for r in runs})
    fig, axes = plt.subplots(len(argvs), 2, figsize=(9, 3.2 * len(argvs)), squeeze=False)
    for row, argv in zip(axes, argvs):
        group = {r['arm']: r for r in runs if r['argv'] == argv}
        for arm, r in sorted(group.items()):
            h = r['hist']
            total = sum(h.values())
            xs, ys, acc = [], [], 0
            for b in range(64):
                acc += h.get(b, 0)
                xs.append(2 ** b); ys.append(acc / total)
            row[0].step(xs, ys, where='post', color=COLORS.get(arm), label=arm)
            med = next(x for x, y in zip(xs, ys) if y >= 0.5)
            fresh = r['rows'][-1]['fresh_addr']
            print(f"{argv:36} {arm:5} reused={total} fresh_addr={fresh} median_distance<= {med}")
        row[0].set_xscale('log', base=2)
        row[0].set_xlim(1, 2 ** 24)
        row[0].set_xlabel('allocations between free and reuse of an address')
        row[0].set_ylabel('CDF')
        row[0].set_title(argv, fontsize=10)
        row[0].legend(fontsize=8, frameon=False)
        if 'off' in group:
            base = [x['win_lines'] for x in group['off']['rows'][2:]]
            for arm, r in sorted(group.items()):
                w = [x['win_lines'] for x in r['rows'][2:]]
                row[1].plot([x['op'] for x in r['rows'][2:]],
                            [a / b for a, b in zip(w, base)], color=COLORS.get(arm), label=arm)
                print(f"{argv:36} {arm:5} mean_lines_per_window/off="
                      f"{sum(w) / sum(base):.3f}")
            row[1].set_xlabel('allocations')
            row[1].set_ylabel('distinct lines handed out / off')
    fig.tight_layout()
    fig.savefig(out + '-reuse.png', dpi=150)


if __name__ == '__main__':
    main()
