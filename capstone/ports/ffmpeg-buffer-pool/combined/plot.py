#!/usr/bin/env python3
"""Publication-sized plots of measured pool behavior, operation counts, and nodes."""
import argparse
import json
import pathlib
import re
import struct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('results', type=pathlib.Path)
parser.add_argument('output', type=pathlib.Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})

def read(p):
    raw = p.read_bytes()
    h = struct.unpack_from('<16Q', raw)
    a = np.frombuffer(raw, dtype='<u8', offset=128).reshape(-1, 16)
    if h[0] != 0x4650465452433032 or h[1] != len(a) or h[2]:
        raise ValueError(f'invalid or failed report: {p}')
    return h, a

header, source = read(args.results / 'recorded.bin')
reports = []
comparisons = []
for mode in range(3):
    h, rows = read(args.results / f'mode-{mode}/capstone.bin')
    exact = np.array_equal(source, rows)
    if not exact: raise ValueError(f'mode {mode} differs')
    comparisons.append(dict(mode=mode, exact=bool(exact), events=len(rows)))
    reports.append(h)
(args.output / 'comparison.json').write_text(json.dumps(comparisons, indent=2) + '\n')
clock = np.cumsum(source[:, 0] == 130)
names = ['Spatial only', 'Backing lifetime', 'Sublet']
colors = ['#567b9b', '#5c9a77', '#c44a32']

fig, axes = plt.subplots(2, 2, figsize=(12.8, 8))
fig.suptitle('FFmpeg: identical pool behavior with lifetime protection', fontsize=16, y=.98)
fig.text(.5, .938, '720p MPEG-4 • 9,000 frames • 20 pools • 135,001 leases • 630,309 events checked', ha='center')
for kind, name, ax in [(1, 'AVBufferPool', axes[0, 0]), (2, 'AVRefStructPool', axes[0, 1])]:
    selected = source[:, 1] == kind
    x = clock[selected]
    rows = source[selected]
    ax.plot(x, rows[:, 12] / 2**20, color='#777777', linewidth=1.7, label='Retained payload')
    ax.plot(x, rows[:, 11] / 2**20, color='#222222', linewidth=1.5, alpha=.7, label='x86 live payload')
    ax.plot(x, rows[:, 11] / 2**20, color=colors[2], linewidth=.9, alpha=.75,
            linestyle='--', label='Sublet live payload')
    ax.set(title=name, xlabel='Completed leases (both pool kinds)', ylabel='Requested payload (MiB)')
    ax.text(.98, .14, 'All three replay arms match every event', transform=ax.transAxes, ha='right', fontsize=9,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(alpha=.15)

ax = axes[1, 0]
for kind, name, color in [(1, 'Buffer pool', '#276c99'), (2, 'Object pool', '#ac5f25')]:
    gaps = source[(source[:, 1] == kind) & (source[:, 0] == 130) & (source[:, 10] > 0), 10]
    vals, counts = np.unique(gaps, return_counts=True)
    ax.step(vals, np.cumsum(counts) / len(gaps), where='post', color=color, label=f'{name}: {len(gaps):,} reuses')
    np.savetxt(args.output / f'reuse-{kind}.csv', np.c_[vals, counts], delimiter=',', header='gap,count', comments='', fmt='%d')
ax.set(xscale='log', ylim=(0, 1.04), title='Reuse-distance distributions are unchanged',
       xlabel='Leases of the same pool kind since last return', ylabel='Cumulative fraction of reuses')
ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=.15)

ax = axes[1, 1]
node_rows = []
for mode in range(3):
    run = pathlib.Path((args.results / f'mode-{mode}/qemu-run-path.txt').read_text().strip())
    log = (run / 'serial.log').read_text()
    matches = re.findall(r'Print = Scalar\(0xff200000([0-9a-f]{8})\).*?REV-NODES alloced_n=(\d+) free_list=(\d+) pool=(\d+)', log, re.S)
    points = []
    for event, allocated, free, capacity in matches:
        index = int(event, 16)
        lease = int(clock[min(index, len(clock)) - 1])
        points.append((lease, int(allocated)))
        node_rows.append((mode, index, lease, int(allocated), int(free), int(capacity)))
    points = np.asarray(points)
    ax.plot(points[:, 0], points[:, 1], color=colors[mode], label=names[mode], linewidth=1.8)
ax.axhline(65536, color='#333333', linestyle=':', linewidth=1)
ax.text(4000, 68000, 'Default budget: 65,536 (original run exhausted it)', fontsize=8)
ax.set(title='QEMU revocation-node allocation grows', xlabel='Completed leases (both pool kinds)', ylabel='Nodes allocated from the pool')
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0)); ax.legend(fontsize=8); ax.grid(alpha=.15)
np.savetxt(args.output / 'nodes.csv', np.asarray(node_rows), delimiter=',',
           header='mode,event,leases,allocated,free_list,capacity', comments='', fmt='%d')
fig.text(.5, .017, 'Payload uses recorded x86 sizes; serial replay executes allocator effects of callbacks. Codec work and metadata are excluded from payload curves.\n'
         'QEMU functional results, no hardware timing claim. Full comparison uses a 1,048,576-node budget; no steady-state reclamation claim.',
         ha='center', fontsize=8, linespacing=1.5)
fig.tight_layout(rect=(0, .07, 1, .925), h_pad=2.5)
for ext in ('png', 'pdf', 'svg'): fig.savefig(args.output / f'ffmpeg-sublet.{ext}', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
labels = ['SPLIT', 'MREV', 'DELIN', 'REVOKE', 'INIT']
x = np.arange(len(labels))
for mode, h in enumerate(reports):
    axes[0].bar(x + (mode - 1) * .25, np.asarray(h[5:10]) / clock[-1] * 1000,
                width=.25, color=colors[mode], label=names[mode])
axes[0].set(xticks=x, xticklabels=labels, ylabel='Operations per 1,000 completed leases', title='Counted lifetime operations')
axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=.15)
axes[1].bar(names, [h[10] / 2**20 for h in reports], color=colors)
axes[1].set(ylabel='MiB written to reinitialize reclaimed regions', title='Reinitialization volume over the whole run')
axes[1].grid(axis='y', alpha=.15)
fig.suptitle('Mechanism costs on the same Capstone allocator port', fontsize=14)
fig.text(.5, .02, 'Counts include trusted callback access and internal pool objects. They are not cycles or FFmpeg throughput.\n'
         'Replay heap high-water: 17,984 B metadata + 10,402,304 B payload in every arm; fixed side tables and observer storage are separate.', ha='center', fontsize=8)
fig.tight_layout(rect=(0, .11, 1, .93))
for ext in ('png', 'pdf', 'svg'): fig.savefig(args.output / f'ffmpeg-costs.{ext}', dpi=180)
plt.close(fig)
