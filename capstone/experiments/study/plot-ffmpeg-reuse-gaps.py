#!/usr/bin/env python3
"""Validate full FFmpeg decoder pool-lease observations and draw paper-width figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARM = {(0, 'capstone'): 'Capstone original',
       (2, 'capstone'): 'Capstone + Sublet',
       (0, 'poisoncap'): 'PoisonCap spatial',
       (2, 'poisoncap'): 'PoisonCap temporal'}
FRAME = re.compile(r'^\s*0,\s*-?\d+,\s*-?\d+,\s*\d+,\s*\d+,\s*([0-9a-f]{32})\s*$', re.M)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_cflags(path):
    lines = [line.removeprefix('CFLAGS=').strip() for line in Path(path).read_text().splitlines()
             if line.startswith('CFLAGS=')]
    opts = [word for word in lines[0].split() if re.fullmatch(r'-O[0-3s]', word)] if len(lines) == 1 else []
    if not opts or opts[-1] != '-O3':
        raise ValueError('FFmpeg library optimization changed')
    return lines[0]


def fields(line):
    return {k: int(v) for k, v in re.findall(r'([a-z_]+)=([0-9]+)', line)}


def parse_gaps(total, pairs):
    if len(total) != 1 or len(pairs) != 16:
        raise ValueError('expected one gap total and 16 bin pairs')
    counts = fields(total[0])
    if set(counts) != {'issues', 'reuses', 'observer'} or counts['observer'] != 16656:
        raise ValueError('wrong gap total or observer size')
    bins = []
    for index, line in enumerate(pairs):
        match = re.fullmatch(r'FF2-GAP pair=(\d+) a=(\d+) b=(\d+)', line)
        if match is None or int(match[1]) != index:
            raise ValueError('missing or unordered gap pair')
        bins.extend([int(match[2]), int(match[3])])
    if sum(bins) != counts['reuses'] or counts['reuses'] > counts['issues']:
        raise ValueError('histogram does not reconcile')
    return dict(issues=counts['issues'], reuses=counts['reuses'], bins=bins,
                observer_bytes=counts['observer'])


def collect(cap_root, poison_root):
    cap_root, poison_root = Path(cap_root), Path(poison_root)
    prior = Path('/tmp/capstone/poisoncap-plots/ffmpeg-poisoncap-matrix')
    cap_image = Path('/tmp/capstone/domain-process-runtime/share/experiments/ffmpeg-reuse.dom')
    poison_image = Path('/tmp/capstone/ffmpeg-reuse-poisoncap-build/ffmpeg')
    clip = Path('/tmp/capstone/domain-process-runtime/share/experiments/clip-1.mkv')
    cap_manifest = json.loads((cap_root/'manifest.json').read_text())
    poison_build = json.loads((poison_image.parent/'manifest.json').read_text())
    if (cap_manifest['images'].get(str(cap_image)) != digest(cap_image) or
        cap_manifest['workload_inputs'].get('/mnt/host/experiments/clip-1.mkv') != digest(clip) or
        poison_build['image_sha256'] != digest(poison_image) or
        poison_build['nested_pool'] != 'poisoncap' or
        not poison_build['pool_reuse_gaps']):
        raise ValueError('measured binary or decoder input changed')
    result_root = Path(__file__).resolve().parent/'results'
    cap_prior = json.loads((result_root/'ffmpeg-pool-memory-20260927/data.json').read_text())['cells']
    poison_prior = json.loads((result_root/'memory-followup-20260927/data.json').read_text())['ffmpeg']
    cap_rows = {}
    for line in (cap_root/'runs.jsonl').read_text().splitlines():
        row = json.loads(line)
        match = re.fullmatch(r'ffmpeg-reuse-b(1|4|16)-pool([02])', row['point']['id'])
        if match is None:
            raise ValueError('unexpected Capstone point')
        label = f'b{match[1]}-m{match[2]}-r{row["repetition"]}'
        cap_rows[label] = row
    poison_rows = {r['label']: r for r in json.loads((poison_root/'runs.json').read_text())}
    if len(cap_rows) != 18 or len(poison_rows) != 18:
        raise ValueError('need 18 independent runs per platform')
    runs = []
    for batches in (1, 4, 16):
        oracle = (prior/f'b{batches}-m0.stdout').read_text()
        expected_frames = FRAME.findall(oracle)
        if len(expected_frames) != 30*batches:
            raise ValueError('bad independent frame oracle')
        for platform, rows, root in (('capstone', cap_rows, cap_root),
                                     ('poisoncap', poison_rows, poison_root)):
            for mode in (0, 2):
                for rep in range(3):
                    label = f'b{batches}-m{mode}-r{rep}'
                    source = rows[label]
                    if platform == 'capstone':
                        folder = root/(source['point']['id']+'-'+str(rep))
                        stdout = (folder/'stdout').read_text()
                        stderr = (folder/'stderr').read_text()
                        if (source['status'] != 'pass' or stdout != oracle or
                            FRAME.findall(stdout) != expected_frames or
                            source['image_sha256'] != digest(cap_image) or
                            digest(folder/'stdout') != source['stdout_sha256'] or
                            digest(folder/'stderr') != source['stderr_sha256']):
                            raise ValueError('Capstone frame or exit oracle failed: '+label)
                        pool = [line for line in stderr.splitlines()
                                if line.startswith('EXP-POOL ')]
                        policy = fields(pool[0]) if len(pool) == 1 else {}
                        if (policy.get('mode') != mode or policy.get('payload') != 315072 or
                            (mode == 0 and policy.get('revoke') != 0) or
                            (mode == 2 and not policy.get('revoke'))):
                            raise ValueError('Capstone pool policy missing: '+label)
                        before = cap_prior[f'b{batches}-m{mode}']['capstone'][rep]
                        release = [line for line in stderr.splitlines()
                                   if line.startswith(f'EXP-MEM phase=released-{batches-1} ')]
                        if (policy != before['pool'] or len(release) != 1 or
                            fields(release[0]) != {k: v for k, v in before['outer_heap'].items()
                                                   if k != 'phase'}):
                            raise ValueError('Capstone memory ledger changed: '+label)
                        rewrite = None
                        total = [line for line in stderr.splitlines()
                                 if line.startswith('FF2-GAP-TOTAL ')]
                        pairs = [line for line in stderr.splitlines()
                                 if line.startswith('FF2-GAP pair=')]
                        raw_sha256 = digest(folder/'stderr')
                    else:
                        stdout = (root/(label+'.stdout')).read_text()
                        stderr = (root/(label+'.stderr')).read_text()
                        if (source['returncode'] or not source['oracle_match'] or
                            stdout != oracle or digest(root/(label+'.stdout')) !=
                            source['stdout_sha256'] or digest(root/(label+'.stderr')) !=
                            source['stderr_sha256']):
                            raise ValueError('PoisonCap frame or exit oracle failed: '+label)
                        policy = [line for line in stderr.splitlines()
                                  if line.startswith('FFPOOL-POLICY ')]
                        if len(policy) != 1 or fields(policy[0]) != {
                            'mode': mode, 'payload_reservation': 4194304}:
                            raise ValueError('PoisonCap policy missing: '+label)
                        adapter = [line for line in stderr.splitlines()
                                   if line.startswith('FF2_POISONCAP ')]
                        if not adapter:
                            raise ValueError('PoisonCap adapter metrics missing: '+label)
                        rewrite = fields(adapter[-1])
                        if mode == 0 and any(rewrite.values()) or mode == 2 and not rewrite['sweeps']:
                            raise ValueError('wrong PoisonCap adapter path: '+label)
                        before = next(x for x in poison_prior if x['batches'] == batches and
                                      x['mode'] == mode and x['variant'] == 'selective')
                        allocated = [int(value) for value in re.findall(
                            r'^EXP-CHERI phase=released-\d+ .*?allocated=(\d+)', stderr, re.M)]
                        if (allocated != before['allocated_at_release'] or
                            rewrite['sweeps'] != before['sweeps'] or
                            rewrite['snapshot_bytes'] != before['snapshot_final'] or
                            (mode == 2 and rewrite['snapshot_peak'] != before['snapshot_peak'])):
                            raise ValueError('PoisonCap memory ledger changed: '+label)
                        total = [line for line in stderr.splitlines()
                                 if line.startswith('FF2-GAP-TOTAL ')]
                        pairs = [line for line in stderr.splitlines()
                                 if line.startswith('FF2-GAP pair=')]
                        raw_sha256 = digest(root/(label+'.stderr'))
                    gap = parse_gaps(total, pairs)
                    runs.append(dict(arm=ARM[mode, platform], platform=platform,
                                     batches=batches, mode=mode, repetition=rep,
                                     frame_count=30*batches, frame_oracle_sha256=
                                     hashlib.sha256(oracle.encode()).hexdigest(),
                                     raw_sha256=raw_sha256, rewrite=rewrite, **gap))
    for batches in (1, 4, 16):
        subset = [r for r in runs if r['batches'] == batches]
        if len({r['issues'] for r in subset}) != 1:
            raise ValueError('different issue counts for matched decoder work')
        if len({tuple(r['bins']) for r in subset}) != 1:
            raise ValueError('pool gap bins differ; revise identical-curves figure')
    return runs


def plot(runs, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.5), sharey=True)
    for ax, batches in zip(axes, (1, 4, 16)):
        r = next(x for x in runs if x['batches'] == batches)
        cdf = np.cumsum(r['bins']) / r['issues']
        x = np.arange(32)
        ax.step(x, cdf, where='post', color='#237a83', lw=1.8)
        ax.scatter(x[:7], cdf[:7], s=9, color='#237a83', zorder=3)
        ax.set_xlim(-.3, 10)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 2, 4, 6, 8, 10], ['1', '4', '16', '64', '256', '1k'])
        ax.set_title(f'{batches} stream'+('s' if batches > 1 else ''))
        ax.set_xlabel('Lease gap (issues)')
        ax.grid(axis='y', alpha=.2)
        ax.text(.96, .08, '4 arms × 3 runs\nidentical bins', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=7, color='#42585b')
    axes[0].set_ylabel('Cumulative reissues / all issues')
    fig.subplots_adjust(left=.095, right=.99, bottom=.22, top=.82, wspace=.15)
    fig.savefig(out/'reuse-gaps.pdf')
    fig.savefig(out/'reuse-gaps.png', dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.05, 2.75))
    x = np.arange(3)
    groups = [next(r for r in runs if r['platform'] == 'poisoncap' and
                   r['mode'] == 2 and r['batches'] == b) for b in (1, 4, 16)]
    bottom = np.zeros(3)
    for key, name, color in (('poison_bytes', 'Poisoned', '#9b5f7a'),
                             ('clear_bytes', 'Cleared', '#d69065'),
                             ('copied_bytes', 'Copied', '#5c8b9a')):
        values = np.array([r['rewrite'][key] for r in groups]) / (1024**2)
        ax.bar(x, values, bottom=bottom, width=.56, label=name, color=color)
        bottom += values
    ax.set_xticks(x, ['1', '4', '16'])
    ax.set_ylabel('Cumulative payload-span bytes (MiB)')
    ax.set_xlabel('Independent 30-frame streams')
    ax.set_title('Selective PoisonCap: explicit per-granule operations and copies')
    ax.legend(frameon=False, ncol=3, loc='upper left')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=.105, right=.99, bottom=.2, top=.86)
    fig.savefig(out/'payload-operations.pdf')
    fig.savefig(out/'payload-operations.png', dpi=250)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--capstone', type=Path)
    source.add_argument('--summary', type=Path)
    p.add_argument('--poisoncap', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.capstone:
        if not args.poisoncap:
            p.error('--capstone requires --poisoncap')
        runs = collect(args.capstone, args.poisoncap)
        args.out.mkdir(parents=True, exist_ok=True)
        document = dict(schema=1, application='FFmpeg 9.0.1',
                        workload='MPEG-4/Matroska decoder, 30 frames per stream',
                        clip_sha256=digest('/tmp/capstone/domain-process-runtime/share/experiments/clip-1.mkv'),
                        builds=dict(
                            capstone_manifest_sha256=digest('/tmp/capstone/ffmpeg-reuse-capstone-sdk-build/manifest.json'),
                            poisoncap_manifest_sha256=digest('/tmp/capstone/ffmpeg-reuse-poisoncap-build/manifest.json'),
                            capstone_lib_config_sha256=digest('/tmp/capstone/application-cheri/pool-capstone/ffbuild/config.mak'),
                            poisoncap_lib_config_sha256=digest('/tmp/capstone/ffmpeg-reuse-poisoncap-build/build/ffbuild/config.mak'),
                            capstone_lib_cflags=config_cflags('/tmp/capstone/application-cheri/pool-capstone/ffbuild/config.mak'),
                            poisoncap_lib_cflags=config_cflags('/tmp/capstone/ffmpeg-reuse-poisoncap-build/build/ffbuild/config.mak')),
                        binaries=dict(capstone=digest('/tmp/capstone/domain-process-runtime/share/experiments/ffmpeg-reuse.dom'),
                            poisoncap=digest('/tmp/capstone/ffmpeg-reuse-poisoncap-build/ffmpeg')),
                        runs=runs)
        (args.out/'summary.json').write_text(json.dumps(document, indent=2)+'\n')
        with (args.out/'bins.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['arm', 'streams', 'repetition', 'log2_gap', 'count'])
            for r in runs:
                for i, count in enumerate(r['bins']):
                    writer.writerow([r['arm'], r['batches'], r['repetition'], i, count])
    else:
        document = json.loads(args.summary.read_text())
        if document['schema'] != 1:
            raise ValueError('unknown summary schema')
        runs = document['runs']
    plot(runs, args.out)
    print(f'Validated {len(runs)} whole decoder runs; four arms, three workloads, three repetitions')


if __name__ == '__main__':
    main()
