#!/usr/bin/env python3
"""Plot the checked output of analyze-reuse.py; no emulator timing metrics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'capstone-sublet': '#087f8c', 'cheribsd-default': '#c55835'}
NAMES = {'capstone-sublet': 'Capstone / Sublet', 'cheribsd-default': 'Default CheriBSD'}
MIB = 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in (args.results/'attempts.jsonl').read_text().splitlines()]
    good = [r for r in rows if r['status']=='pass']
    required = {('ffmpeg',30,n,0) for n in (1,4,16,64)} | {
        ('mruby',32,8,256), ('mruby',128,8,256), ('mruby',512,8,256),
        ('mruby',128,8,4096), ('mruby',512,16,256), ('mruby',128,32,4096)}
    for key in required:
        for arm in COLORS:
            if sum(tuple(r['workload'])==key and r['arm']==arm for r in good)!=3:
                raise ValueError(f'three passing repeats required for plot: {key}, {arm}')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.dpi': 180})

    def save(fig, name, note):
        fig.tight_layout(rect=(0, .13, 1, .93))
        fig.text(.5, .025, note, ha='center', va='bottom', fontsize=8)
        for suffix in ('pdf', 'png'):
            fig.savefig(args.out/(name+'.'+suffix), bbox_inches='tight')
        plt.close(fig)

    def curve(ax, values, arm, scale=1):
        x = sorted(values)
        ax.plot(x, [median(values[i])/scale for i in x], color=COLORS[arm],
                label=NAMES[arm], marker='o', markersize=3)
        ax.fill_between(x, [min(values[i])/scale for i in x],
                        [max(values[i])/scale for i in x], color=COLORS[arm], alpha=.18)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, key, title in zip(axes, [('ffmpeg',30,64,0), ('mruby',512,16,256)],
                              ['FFmpeg: 64 independent streams', 'mruby: 512 records, 16 batches']):
        for arm in COLORS:
            values = defaultdict(list)
            for r in good:
                if tuple(r['workload'])!=key or r['arm']!=arm: continue
                for i, k in enumerate(('reuse1', 'reuse8', 'reuse64', 'reuse512', 'reuse4096', 'reuse_more')):
                    values[i].append(100*r['metrics'][k+'_fraction'])
            curve(ax, values, arm)
        ax.set_title(title); ax.set_xticks(range(6), ['1','8','64','512','4096','any'])
        ax.set_xlabel('Allocation-call distance since the last free')
        ax.set_ylabel('Allocations reusing a start address (%)')
        ax.set_ylim(-2,102); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Prompt reuse of freed allocation addresses')
    save(fig, 'reuse-window', 'All successful allocation calls are the denominator; in-place realloc is not reuse.\nThree runs per point; median and full range. These are call distances, not time or cache measurements.')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for arm in COLORS:
        values = defaultdict(list)
        for r in good:
            app,size,batches,keep = r['workload']
            if r['arm']==arm and app=='ffmpeg' and size==30:
                values[batches].append(r['metrics']['unique'])
        curve(axes[0], values, arm)
        values = defaultdict(list)
        for r in good:
            app,size,batches,keep = r['workload']
            if r['arm']==arm and app=='mruby' and batches==8 and keep==256:
                values[size].append(r['metrics']['unique'])
        curve(axes[1], values, arm)
    axes[0].set_title('FFmpeg: repeated 30-frame streams')
    axes[0].set_xlabel('Completed streams'); axes[0].set_xscale('log', base=2)
    axes[0].set_xticks([1,4,16,64], ['1','4','16','64'])
    axes[1].set_title('mruby: 8 batches, retained graph of 256')
    axes[1].set_xlabel('Records per ordinary batch'); axes[1].set_xscale('log', base=2)
    axes[1].set_xticks([32,128,512], ['32','128','512'])
    for ax in axes:
        ax.set_ylabel('Distinct allocation start addresses'); ax.set_ylim(bottom=0)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Address history required by repeated application work')
    save(fig, 'address-history', 'Three runs per point; median and full range. All instrumented calls, no event replay.\nDistinct starts do not measure distinct pages, resident memory, cache misses or external fragmentation.')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    keys = [('ffmpeg',30,64,0), ('mruby',512,16,256), ('mruby',128,32,4096)]
    titles = ['FFmpeg: 64 streams', 'mruby: 16 batches, small retained graph', 'mruby: 32 batches, large retained graph']
    for ax, key, title in zip(axes, keys, titles):
        for arm in COLORS:
            values = defaultdict(list)
            for r in good:
                if tuple(r['workload'])!=key or r['arm']!=arm: continue
                for phase in r['phases']:
                    if phase['phase'].startswith('released-'):
                        values[int(phase['phase'].split('-')[1])+1].append(phase['ledger'])
            curve(ax, values, arm, MIB)
        if key[0]=='mruby': ax.axvline(key[2]//2+1, color='#999999', ls=':', label='4× burst')
        ax.set_title(title); ax.set_xlabel('Completed stream / request batch')
        ax.set_ylabel('Post-release allocator ledger (MiB)'); ax.set_ylim(bottom=0)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Storage still charged to the allocator after each release')
    save(fig, 'release-retention', 'Capstone: occupied buddy blocks. CheriBSD: jemalloc allocated, including quarantine and private allocations.\nNeither is total RSS. Capstone still reserves 8 MiB / 16 MiB logical pools from 16 MiB / 32 MiB grants, respectively.\nThree runs per point; median and full range. Gray lines mark mruby’s burst; no forced revocation or policy overrides.')

    fig, ax = plt.subplots(figsize=(8, 4.6))
    keys = [('mruby',32,8,256), ('mruby',128,8,256), ('mruby',512,8,256), ('mruby',128,8,4096)]
    for j, arm in enumerate(COLORS):
        med, low, high = [], [], []
        for key in keys:
            a = [r['metrics']['released_ledger_last']/MIB for r in good
                 if tuple(r['workload'])==key and r['arm']==arm]
            mid=median(a); med.append(mid); low.append(mid-min(a)); high.append(max(a)-mid)
        ax.bar([i+(j-.5)*.36 for i in range(len(keys))], med, width=.36,
               color=COLORS[arm], label=NAMES[arm], yerr=[low,high], capsize=3)
    ax.set_xticks(range(4), ['32 / 256', '128 / 256', '512 / 256', '128 / 4096'])
    ax.set_xlabel('Ordinary records / retained records'); ax.set_ylabel('Final post-release ledger (MiB)')
    ax.grid(axis='y',alpha=.2); ax.legend(); fig.suptitle('mruby: retention advantage depends on the live working set')
    save(fig, 'retained-graph-tradeoff', 'Eight batches with a 4× burst, three runs per point; median and full range.\nOccupied buddy blocks versus jemalloc allocated: different ledgers, neither total RSS.\nCapstone has a 16 MiB logical pool / 32 MiB grant; free pool capacity remains reserved.')


if __name__ == '__main__':
    main()
