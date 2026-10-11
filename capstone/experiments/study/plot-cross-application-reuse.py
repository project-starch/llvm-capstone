#!/usr/bin/env python3
"""Render checked, complete-application inner-allocator reuse observations.

This preview has three qualified boundaries. It never fills unqualified arms
or treats allocation events as independent process repetitions.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / 'results'
ARMS = ('capstone-original', 'capstone-sublet',
        'poisoncap-spatial', 'poisoncap-temporal')
STYLE = {
    'capstone-original': ('Capstone original', '#0072B2', '--'),
    'capstone-sublet': ('Capstone + Sublet', '#0072B2', '-'),
    'poisoncap-spatial': ('CheriBSD spatial', '#D55E00', '--'),
    'poisoncap-temporal': ('PoisonCap', '#D55E00', '-'),
}


def records():
    sqlite = json.loads((ROOT/'sqlite-reuse-gaps-20260927/runs.json').read_text())
    mruby = json.loads((ROOT/'mruby-gc-memory-20260927/summary.json').read_text())['runs']
    ffmpeg = json.loads((ROOT/'ffmpeg-reuse-gaps-20260927/summary.json').read_text())['runs']
    for row in sqlite:
        arm = {'capstone': ARMS[0], 'capstone-sublet': ARMS[1],
               'poisoncap-spatial': ARMS[2], 'poisoncap-temporal': ARMS[3]}[row['arm']]
        yield 'SQLite 3.22.0 · memsys5', arm, row['rep'], row['allocations'], row['reuses'], row['bins']
    for row in mruby:
        arm = {'capstone-spatial': ARMS[0], 'capstone-sublet': ARMS[1],
               'poisoncap-spatial': ARMS[2], 'poisoncap-temporal': ARMS[3]}[row['arm']]
        phase = row['phases'][-1]
        yield 'mruby 4.0.0-rc2 · GC slots', arm, row['rep'], phase['issues'], phase['reissues'], row['bins']
    for row in ffmpeg:
        if row['batches'] != 16:
            continue
        arm = {'Capstone original': ARMS[0], 'Capstone + Sublet': ARMS[1],
               'PoisonCap spatial': ARMS[2], 'PoisonCap temporal': ARMS[3]}[row['arm']]
        yield 'FFmpeg 9.0.1 · pool leases', arm, row['repetition'], row['issues'], row['reuses'], row['bins']


def checked():
    cells = defaultdict(list)
    for application, arm, rep, issues, reuses, bins in records():
        if len(bins) != 32 or issues <= 0 or sum(bins) != reuses or reuses > issues:
            raise ValueError(f'invalid release-gap accounting: {application}, {arm}, {rep}')
        cells[application, arm].append((rep, issues, reuses, bins))
    names = ('SQLite 3.22.0 · memsys5', 'mruby 4.0.0-rc2 · GC slots',
             'FFmpeg 9.0.1 · pool leases')
    if set(cells) != {(name, arm) for name in names for arm in ARMS}:
        raise ValueError('three complete four-arm application boundaries required')
    result = {}
    for name in names:
        result[name] = {}
        for arm in ARMS:
            rows = sorted(cells[name, arm])
            if len(rows) != 3 or len({r[0] for r in rows}) != 3:
                raise ValueError(f'three independent processes required: {name}, {arm}')
            if any(r[1:] != rows[0][1:] for r in rows[1:]):
                raise ValueError(f'repetitions diverge; show their range: {name}, {arm}')
            _, issues, reuses, bins = rows[0]
            result[name][arm] = {'issues': issues, 'reuses': reuses,
                                 'bins': bins, 'repetitions': 3}
    return result


def draw(data, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    names = tuple(data)
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.85), sharey=True,
                             gridspec_kw={'wspace': .16})
    limits = (2**20, 2**18, 2**7)
    for ax, name, xmax in zip(axes, names, limits):
        for arm in ARMS:
            row = data[name][arm]
            cumulative = 0
            values = [0.]
            for count in row['bins']:
                cumulative += count
                values.append(100*cumulative/row['issues'])
            x = [1] + [2**(i+1)-1 for i in range(32)]
            label, color, line = STYLE[arm]
            ax.step(x, values, where='post', label=label, color=color,
                    linestyle=line, linewidth=1.6 if line == '-' else 1.25,
                    zorder=3 if line == '-' else 2)
        ax.set_xscale('log', base=2)
        ax.set_xlim(1, xmax)
        ax.set_ylim(0, 102)
        ax.set_title(name, fontsize=8.2, pad=7)
        ax.set_xlabel('Allocations since release', fontsize=7.5)
        ax.grid(axis='y', color='#d9d9d9', linewidth=.55)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel('Reused start / all issues (%)', fontsize=7.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(.5, -.015), fontsize=7.4)
    fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.30)
    fig.savefig(output/'reuse-cdf.pdf')
    fig.savefig(output/'reuse-cdf.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = checked()
    (args.out/'data.json').write_text(json.dumps(data, indent=2)+'\n')
    draw(data, args.out)


if __name__ == '__main__':
    main()
