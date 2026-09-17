#!/usr/bin/env python3
"""Scientific plot of independently observed memory behavior, never timings."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ffpool-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('results', type=Path)
a = p.parse_args()
spec = importlib.util.spec_from_file_location('trace_tools', Path(__file__).with_name('trace-tools.py'))
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)
root = a.results
comparison = json.loads((root / 'comparison.json').read_text())
if any(r['mismatching_events'] for r in comparison['comparisons'].values()):
    raise SystemExit('plot requires successful event-by-event comparison')
workload = json.loads((root / 'workload.json').read_text())
arms = [('x86 FFmpeg recording', 'original.bin', '#aaaaaa', 5.5),
        ('x86 component replay', 'native.bin', '#e69f00', 2.7),
        ('Capstone / QEMU replay', 'capstone.bin', '#0072b2', 1.2)]
data = [(label, np.array(trace.read(root / file)[0], dtype=np.uint64), color, width)
        for label, file, color, width in arms]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.titleweight': 'bold'})
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5), gridspec_kw={'width_ratios': [1, 1.25, 1.15]})
stats = comparison['original']
max_gap = stats['max_gap']
edges = np.array([2**k - 1 for k in range(1, max(4, max_gap.bit_length() + 2))])
for label, rows, color, width in data:
    gets = rows[rows[:, 0] == 2]
    gaps = gets[:, 7]
    cdf = [np.count_nonzero((gaps > 0) & (gaps <= x)) / len(gets) for x in edges]
    axes[0].step(edges, cdf, where='post', color=color, linewidth=width, label=label)
    if label.startswith('Capstone'):
        axes[0].plot(edges, cdf, 'o', color=color, markersize=4)
    index = np.arange(len(rows))
    for ax, limit in ((axes[1], len(rows)), (axes[2], min(110, len(rows)))):
        x = index[:limit]
        if ax is axes[1]: x = x / 1000
        ax.plot(x, rows[:limit, 9] / 2**20, color=color, linewidth=width,
                solid_capstyle='round')
        ax.plot(x, rows[:limit, 8] / 2**20, color=color, linewidth=width,
                linestyle='--', solid_capstyle='round')

axes[0].set(title='(a) Free-to-reuse distance', xscale='log', ylim=(0, 1.055),
            xlabel='Pool allocations since final return', ylabel='Fraction of all pool allocations')
axes[0].set_xscale('log', base=2)
axes[0].set_xticks(edges)
axes[0].xaxis.set_major_formatter(ScalarFormatter())
axes[0].text(.97, .1, f"{stats['reuses']:,} / {stats['allocations']:,} reused\n"
             f"{100 * stats['reuse_fraction']:.3f}%", ha='right', transform=axes[0].transAxes)
axes[1].set(title='(b) Complete workload', xlabel='Pool event index (thousands)',
            ylabel='Pool payload (MiB)')
axes[2].set(title='(c) Warm-up and first reuses', xlabel='Pool event index',
            ylabel='Pool payload (MiB)')
for ax in axes:
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
for ax in axes[1:]:
    ax.set_ylim(bottom=0)
axes[2].legend(handles=[Line2D([], [], color='black', label='Retained by pools'),
                       Line2D([], [], color='black', linestyle='--', label='Issued to callers')],
               fontsize=8, loc='lower right', framealpha=.95)
fig.suptitle(f"FFmpeg: {workload['frames']:,} frames · {workload['dimensions']} · MPEG-4 decode", y=.985, fontsize=14)
fig.legend(handles=[Line2D([], [], color=c, linewidth=w, label=l) for l, _, c, w in arms],
           loc='upper center', bbox_to_anchor=(.5, .935), ncol=3, frameon=False)
fig.text(.5, .035, f"{stats['events']:,} events: zero differences in backing identity, reuse gap, or payload bytes. "
         "Both replays are serial and unprotected.", ha='center', fontsize=9)
fig.subplots_adjust(top=.77, bottom=.22, left=.055, right=.985, wspace=.32)
for suffix in ('png', 'pdf', 'svg'):
    fig.savefig(root / f'ffmpeg-replay.{suffix}', dpi=180, facecolor='white')
print(root / 'ffmpeg-replay.png')
