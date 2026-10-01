#!/usr/bin/env python3
"""Draw the checked AO width-8/16 GC reuse and page-group comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS = ('capstone-spatial', 'capstone-sublet',
        'poisoncap-spatial', 'poisoncap-temporal')
LABELS = ('Capstone spatial', 'Capstone + Sublet',
          'PoisonCap spatial', 'PoisonCap temporal')
COLORS = ('#777777', '#1676aa', '#b17d49', '#b33b5a')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_summary(path, width):
    source = json.loads(path.read_text())
    if source['benchmark'] != f'mruby 4.0.0-rc2 bm_ao_render.rb width={width}':
        raise ValueError('wrong AO work unit')
    runs = source['runs']
    if len(runs) != 12 or {(r['arm'], r['rep']) for r in runs} != \
            {(arm, rep) for arm in ARMS for rep in range(3)}:
        raise ValueError('incomplete repeated four-arm matrix')
    if len({r['workload_sha256'] for r in runs}) != 1:
        raise ValueError('workload differs across arms')
    result = {}
    for arm in ARMS:
        samples = [r for r in runs if r['arm'] == arm]
        metric = lambda r: (r['phases'][2]['issues'], r['gap_le_1023'],
                            r['phases'][2]['peak_pages'], r['phases'][2]['pages'],
                            r['selected_peak_bytes'], r['selected_after_bytes'])
        if len({metric(r) for r in samples}) != 1 or \
                len({r['binary_sha256'] for r in samples}) != 1:
            raise ValueError('replicates differ; use a variation-aware plot')
        first = samples[0]
        result[arm] = dict(issues=first['phases'][2]['issues'],
                           gap_le_1023=first['gap_le_1023'],
                           peak_pages=first['phases'][2]['peak_pages'],
                           after_pages=first['phases'][2]['pages'],
                           selected_peak_bytes=first['selected_peak_bytes'],
                           selected_after_bytes=first['selected_after_bytes'],
                           binary_sha256=first['binary_sha256'])
    return result, next(iter({r['workload_sha256'] for r in runs}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--width8', type=Path, required=True)
    p.add_argument('--width16', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    data8, work8 = read_summary(args.width8, 8)
    data16, work16 = read_summary(args.width16, 16)
    if work8 != work16 or any(data8[a]['binary_sha256'] != data16[a]['binary_sha256']
                              for a in ARMS):
        raise ValueError('workload or interpreter binary changed with AO size')
    data = {8: data8, 16: data16}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'scale-summary.json').write_text(json.dumps(dict(
        benchmark='mruby 4.0.0-rc2 upstream AO width 8 and 16',
        workload_sha256=work8,
        summary_sha256={'8': digest(args.width8), '16': digest(args.width16)},
        cells=data), indent=2)+'\n')
    with (args.out/'scale.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('width', 'arm', 'issues', 'gap_le_1023',
                         'gap_le_1023_percent', 'peak_groups', 'after_groups',
                         'selected_peak_bytes', 'selected_after_bytes'))
        for width in (8, 16):
            for arm in ARMS:
                cell = data[width][arm]
                writer.writerow((width, arm, cell['issues'], cell['gap_le_1023'],
                    f"{100*cell['gap_le_1023']/cell['issues']:.6f}",
                    cell['peak_pages'], cell['after_pages'],
                    cell['selected_peak_bytes'], cell['selected_after_bytes']))
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), layout='constrained')
    x = np.arange(2)
    for i, (arm, label, color) in enumerate(zip(ARMS, LABELS, COLORS)):
        offset = (i-1.5)*.19
        axes[0].bar(x+offset, [data[w][arm]['gap_le_1023'] /
                               data[w][arm]['issues'] * 100 for w in (8, 16)],
                    .18, color=color, label=label)
        axes[1].bar(x+offset, [data[w][arm]['peak_pages'] for w in (8, 16)],
                    .18, color=color)
    axes[0].set_ylabel('Reissues within 1,023 / issues (%)')
    axes[1].set_ylabel('Peak GC page groups')
    for ax in axes:
        ax.set_xticks(x, ['width 8', 'width 16'])
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(0, max(data[w][a]['peak_pages']
                            for w in (8, 16) for a in ARMS)*1.2)
    fig.legend(loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(.5, 1.07), fontsize=7)
    for ext in ('pdf', 'png'):
        fig.savefig(args.out/f'gc-size-effect.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
