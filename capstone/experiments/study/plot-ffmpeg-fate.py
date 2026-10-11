#!/usr/bin/env python3
"""Render checked adapted-FATE pool reuse and selective snapshot backing."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CASES = ('xvid_vlc_trac7411', 'resize_down-up')
ARMS = ('capstone-pool0', 'capstone-pool2',
        'poisoncap-spatial', 'poisoncap-temporal')
LABELS = ('Capstone original', 'Capstone + Sublet',
          'PoisonCap spatial', 'PoisonCap temporal')
COLORS = ('#69859a', '#237a83', '#aa8d80', '#a4566a')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    runs = data['runs']
    if data['schema'] != 1 or data['valid_attempts'] != 24 or len(runs) != 24:
        raise ValueError('incomplete FATE qualification')
    expected = {(case, arm, rep) for case in CASES for arm in ARMS for rep in range(3)}
    if {(r['case'], r['arm'], r['repetition']) for r in runs} != expected:
        raise ValueError('missing or duplicate FATE process')

    reuse = np.zeros((len(ARMS), len(CASES)))
    for i, arm in enumerate(ARMS):
        for j, case in enumerate(CASES):
            group = [r for r in runs if r['arm'] == arm and r['case'] == case]
            values = {sum(r['bins'][:4]) / r['issues'] for r in group}
            if len(values) != 1:
                raise ValueError('repetition mismatch')
            reuse[i, j] = values.pop() * 100
    for j in range(len(CASES)):
        if len(set(reuse[:, j])) != 1:
            raise ValueError('different reuse bins: redraw this figure')

    peaks = []
    for case in CASES:
        values = {r['adapter']['snapshot_peak'] for r in runs
                  if r['case'] == case and r['arm'] == 'poisoncap-temporal'}
        if len(values) != 1:
            raise ValueError('snapshot peak differs across repetitions')
        peaks.append(values.pop() / 1024)

    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.0))
    x = np.arange(len(CASES))
    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        axes[0].bar(x + (i - 1.5) * .18, reuse[i], width=.17,
                    label=label, color=color)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel('Reissued within 15 leases / all issues (%)')
    axes[0].text(.97, .98, '4 arms: identical bins', ha='right', va='top',
                 transform=axes[0].transAxes, color='#43515b')
    axes[1].bar(x, peaks, width=.52, color=COLORS[-1])
    axes[1].set_ylim(0, max(peaks) * 1.28)
    axes[1].set_ylabel('Peak snapshot backing (KiB)')
    for j, peak in enumerate(peaks):
        axes[1].text(j, peak + 5, f'{peak:.1f}', ha='center', va='bottom')
    for axis in axes:
        axis.set_xticks(x, ['Xvid, 20 frames', 'Resize, 150 frames'])
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
               ncol=4, frameon=False, bbox_to_anchor=(.5, .12), fontsize=7)
    fig.subplots_adjust(left=.105, right=.985, top=.94, bottom=.37, wspace=.39)
    fig.text(.105, .045, 'Adapted FFmpeg 9.0.1 FATE inputs; 3 exact-oracle processes per arm and input.\n'
             'Snapshot backing is a selected adapter component, not total memory.', fontsize=7)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / 'fate-pool-memory.pdf', bbox_inches='tight')
    fig.savefig(args.out / 'fate-pool-memory.png', dpi=250, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
