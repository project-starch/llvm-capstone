#!/usr/bin/env python3
"""Validate mruby AO GC-slot campaigns and draw memory-behaviour figures."""
import argparse
import csv
import hashlib
import json
import re
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
ORACLES = {8: ('28060a9790593ace825b7552ab5d1878cf3e95be3be08ee738f28cde4cee5d11', 203),
           16: ('cfd73bb1de97514b2adacbda3a377e5172aecbe48819f654071d1996359ca634', 781)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(stderr):
    rows = []
    bins = None
    for line in stderr.splitlines():
        if line.startswith('MRB_GC_STUDY '):
            row = dict(word.split('=', 1) for word in line.split()[1:])
            rows.append({k: v if k == 'phase' else int(v) for k, v in row.items()})
        if line.startswith('MRB_GC_GAPS '):
            if bins is not None:
                raise ValueError('duplicate GC histogram')
            fields = dict(word.split('=', 1) for word in line.split()[1:])
            if set(fields) != {f'b{i}' for i in range(32)}:
                raise ValueError('malformed GC histogram')
            bins = [int(fields[f'b{i}']) for i in range(32)]
    if [r['phase'] for r in rows] != ['startup', 'before', 'after', 'exit'] or bins is None:
        raise ValueError('missing phase or histogram')
    last = rows[-1]
    if sum(bins) != last['reissues'] or not (last['issues'] >= last['releases'] >=
            last['reissues'] >= last['gap15']) or rows[2]['peak_pages'] < rows[2]['pages']:
        raise ValueError('GC accounting does not reconcile')
    return rows, bins


def collect(capstone, poisoncap, width=16):
    runs = []
    oracle, output_bytes = ORACLES[width]
    workload_hashes = set()
    for platform, root in (('capstone', Path(capstone)), ('poisoncap', Path(poisoncap))):
        manifest = json.loads((root/'manifest.json').read_text())
        workload_sha = (manifest['workload_inputs']['/mnt/host/study-mruby/ao.rb']
                        if platform == 'capstone' else
                        manifest['files']['/tmp/ao.rb']['sha256'])
        workload_hashes.add(workload_sha)
        raw = [json.loads(line) for line in (root/'runs.jsonl').read_text().splitlines()]
        if len(raw) != 6:
            raise ValueError(f'{platform}: expected six attempts in one guest')
        if platform == 'capstone' and len({r['boot_id'] for r in raw}) != 1:
            raise ValueError('Capstone guest rebooted during campaign')
        if platform == 'capstone' and any(r['after']['live_domains'] or
                r['after']['live_regions'] or r['after']['nodes_high_water'] >=
                r['after']['node_capacity'] for r in raw):
            raise ValueError('Capstone cleanup or node capacity failed')
        for record in raw:
            point = record['point']
            arm = point['arm']
            marker = '-r' if platform == 'capstone' else '-'
            match = re.fullmatch(r'ao'+str(width)+'-'+re.escape(arm)+marker+r'([0-2])', point['id'])
            if arm not in ARMS or not arm.startswith(platform) or not match or \
                    record['status'] != 'pass' or record['repetition'] != 0 or \
                    point['argv'][-1] != str(width) or point['expected_stdout_sha256'] != oracle:
                raise ValueError('unexpected or failed mruby AO point')
            rep = int(match[1])
            directory = root/(point['id']+'-0')
            stdout, stderr = directory/'stdout', directory/'stderr'
            if digest(stdout) != oracle or stdout.stat().st_size != output_bytes or \
                    record['stdout_sha256'] != oracle or \
                    record['stderr_sha256'] != digest(stderr):
                raise ValueError('AO binary oracle or transcript hash mismatch')
            if platform == 'capstone':
                image = point['image']
                if manifest['images'][image] != record['image_sha256']:
                    raise ValueError('Capstone image changed')
            else:
                if manifest['files']['/tmp/mruby-gc']['sha256'] != digest(
                        next(host for host, guest in point['files'].items()
                             if guest == '/tmp/mruby-gc')):
                    raise ValueError('PoisonCap image changed')
                if record['effective_environment']['MRB_GC_POISONCAP'] != str(int(arm.endswith('temporal'))):
                    raise ValueError('PoisonCap mode mismatch')
            rows, bins = metrics(stderr.read_text(errors='replace'))
            if any(row['mode'] != int(arm in ('capstone-sublet', 'poisoncap-temporal'))
                   for row in rows):
                raise ValueError('GC mode mismatch')
            if rows[-1]['pages'] or rows[-1].get('quarantine', 0):
                raise ValueError('GC resources retained at process exit')
            after = rows[2]
            if platform == 'capstone' and arm.endswith('sublet'):
                inner = record['inner_memory']
                if [r['phase'] for r in inner] != ['startup', 'before', 'after', 'exit'] or \
                        inner[2]['issues'] != after['issues'] or \
                        inner[2]['revoke'] != after['releases']:
                    raise ValueError('Sublet slot operations do not reconcile')
            if platform == 'poisoncap':
                end = rows[-1]
                if arm.endswith('spatial'):
                    if any(end[k] for k in ('sweeps', 'poison_bytes', 'clear_bytes',
                                            'zero_bytes', 'discarded_quarantine')):
                        raise ValueError('spatial control performed temporal operations')
                elif not (end['sweeps'] > 0 and end['poison_bytes'] ==
                          end['releases']*end['slot_bytes'] and
                          end['clear_bytes'] == end['zero_bytes'] and
                          end['clear_bytes'] + end['discarded_quarantine']*end['slot_bytes'] ==
                          end['poison_bytes']):
                    raise ValueError('PoisonCap poison/clear/quarantine accounting does not reconcile')
            if after['page_payload_bytes'] != 1024*after['slot_bytes'] or \
                    after['page_metadata_bytes'] < after['observer_bytes']:
                raise ValueError('GC page layout does not reconcile')
            page_bytes = (after['page_payload_bytes'] + after['page_metadata_bytes']
                          if platform == 'capstone' else after['mapped_page_bytes'])
            if page_bytes < after['page_payload_bytes'] + after['page_metadata_bytes']:
                raise ValueError('GC mapping is shorter than its page structure')
            runs.append(dict(platform=platform, arm=arm, rep=rep,
                             workload_sha256=workload_sha,
                             stdout_sha256=oracle, stderr_sha256=digest(stderr),
                             binary_sha256=(record['image_sha256'] if platform == 'capstone'
                                            else manifest['files']['/tmp/mruby-gc']['sha256']),
                             phases=rows, bins=bins,
                             selected_page_bytes=page_bytes,
                             selected_peak_bytes=page_bytes*after['peak_pages'],
                             selected_after_bytes=page_bytes*after['pages'],
                             gap_le_1023=sum(bins[:10])))
    if len(workload_hashes) != 1:
        raise ValueError('Capstone and PoisonCap ran different AO sources')
    if {(r['arm'], r['rep']) for r in runs} != {(a, i) for a in ARMS for i in range(3)}:
        raise ValueError('incomplete four-arm matrix')
    issues = {r['phases'][-1]['issues'] for r in runs}
    if len(issues) != 1:
        raise ValueError('different GC allocation demand across arms')
    for arm in ARMS:
        cells = [r for r in runs if r['arm'] == arm]
        signature = lambda r: (tuple(r['bins']), r['phases'][2]['pages'],
                               r['phases'][2]['peak_pages'], r['phases'][-1]['gap15'])
        if len({signature(r) for r in cells}) != 1:
            raise ValueError('within-arm GC behavior differs; plot replicate variation explicitly')
    return sorted(runs, key=lambda r: (ARMS.index(r['arm']), r['rep']))


def draw(runs, out):
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(5.2, 2.55), layout='constrained')
    for arm, label, color in zip(ARMS, LABELS, COLORS):
        sample = next(r for r in runs if r['arm'] == arm)
        cumulative = np.cumsum(sample['bins']) / sample['phases'][-1]['issues'] * 100
        ax.step(np.arange(1, 33), cumulative, where='post', label=label,
                linewidth=1.7, color=color)
    ax.set(xlim=(0, 19), ylim=(0, 100), xlabel='Release-to-reissue gap (issues; log₂ scale)',
           ylabel='Cumulative reissues / issues (%)')
    ax.set_xticks([1, 4, 8, 12, 16], ['1', '15', '255', '4,095', '65,535'])
    ax.grid(axis='y', alpha=.22)
    ax.legend(loc='lower right', frameon=False, fontsize=7)
    for ext in ('pdf', 'png'):
        fig.savefig(out/f'gc-reuse-cdf.{ext}', dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 2.4), layout='constrained')
    values = [next(r for r in runs if r['arm'] == arm)['phases'][2]['peak_pages']
              for arm in ARMS]
    after = [next(r for r in runs if r['arm'] == arm)['phases'][2]['pages']
             for arm in ARMS]
    x = np.arange(4)
    ax.bar(x-.18, values, .34, color=COLORS, alpha=.95, label='Peak during render')
    ax.bar(x+.18, after, .34, color=COLORS, alpha=.4, label='After render')
    ax.set_xticks(x, LABELS, rotation=15, ha='right')
    ax.set_ylabel('GC page groups')
    ax.set_ylim(0, max(values)*1.2)
    ax.grid(axis='y', alpha=.22)
    ax.legend(frameon=False, fontsize=7)
    for ext in ('pdf', 'png'):
        fig.savefig(out/f'gc-page-groups.{ext}', dpi=240)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capstone', type=Path)
    p.add_argument('--poisoncap', type=Path)
    p.add_argument('--summary', type=Path, help='Redraw from validated summary only')
    p.add_argument('--width', type=int, choices=sorted(ORACLES), default=16)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.summary:
        source = json.loads(args.summary.read_text())
        runs, benchmark = source['runs'], source['benchmark']
    elif args.capstone and args.poisoncap:
        runs = collect(args.capstone, args.poisoncap, args.width)
        benchmark = f'mruby 4.0.0-rc2 bm_ao_render.rb width={args.width}'
    else:
        p.error('provide both raw campaign roots or --summary')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps({'benchmark': benchmark,
                                                     'runs': runs}, indent=2)+'\n')
    with (args.out/'bins.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('arm', 'rep', 'log2_bin', 'reissues'))
        for r in runs:
            for i, count in enumerate(r['bins']):
                writer.writerow((r['arm'], r['rep'], i, count))
    draw(runs, args.out)


if __name__ == '__main__':
    main()
