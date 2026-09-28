#!/usr/bin/env python3
"""Paper figures from the complete SQLite, mruby and FFmpeg campaigns.

No guest execution or workload replay. Preserve every admitted workload in
the supplement; compare protection with its own spatial control. Revalidate
the committed SQLite raw ledgers and the other campaigns' checked summaries.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'results'
ARMS = ('capstone-spatial', 'capstone-sublet',
        'poisoncap-spatial', 'poisoncap-temporal')
PAIRS = ((ARMS[0], ARMS[1]), (ARMS[2], ARMS[3]))
BLUE, ORANGE = '#0072B2', '#D55E00'
COLORS = (BLUE, BLUE, ORANGE, ORANGE)
MARKERS = ('x', 'o', '+', 's')
LABELS = ('Capstone spatial', 'Sublet', 'PoisonCap spatial', 'PoisonCap temporal')
EDGES = np.array([2**(i+1)-1 for i in range(32)])
MAIN = ('sqlite-main', 'mruby-ao16', 'ffmpeg-resize')
TITLES = {
    'sqlite-main': 'SQLite · memsys5\nspeedtest1 main',
    'mruby-ao8': 'mruby · GC slots\nAO, width 8',
    'mruby-ao16': 'mruby · GC slots\nAO, width 16',
    'ffmpeg-xvid': 'FFmpeg · pool leases\nFATE Xvid, 20 frames',
    'ffmpeg-resize': 'FFmpeg · pool leases\nFATE resize, 150 frames',
    **{f'ffmpeg-streams{n}': f'FFmpeg · pool leases\n{n} × 30-frame stream'
       for n in (1, 4, 16)},
}
CAPTIONS = {
    '01-reuse-and-control': (
        'Reuse in complete applications: SQLite 3.22.0 speedtest1 main (17 units, size 1), '
        'mruby 4.0.0-rc2 AO (width 16), and FFmpeg 9.0.1 (adapted FATE resolution-change input). '
        'Top: cumulative same-start reissues divided by all issues, at log-bin upper edges; '
        'immediate reuse has gap one. Bottom: each spatial control minus its protected arm, '
        'in percentage points. Positive values mean less prompt reuse with protection. '
        'Three processes per arm have identical bins. Sublet preserves SQLite reuse and '
        'closely tracks mruby; all four FFmpeg curves coincide. These are allocation gaps, '
        'not elapsed time, physical working set or total memory. All eight inputs appear in Figure 5.'),
    '02-sqlite-memory': (
        'SQLite 3.22.0, repeated complete speedtest1 main workloads, size 1 and lookaside disabled. '
        'Allocator state persists across one warmup (shaded) and 16 measured units; each unit '
        'opens and closes a database. Both metrics are divided by their own platform’s '
        'original-layout control at the same unit. Address coverage is a cumulative interval union. '
        'Selected bytes are the exact within-unit peak of live plus quarantined spans and allocator '
        'tables; platform node/kernel storage is excluded. Three repetitions coincide. Sublet '
        'avoids address expansion but has the higher selected peak-byte ratio. PoisonCap uses '
        'the corrected quarantine path.'),
    '03-mruby-memory': (
        'mruby 4.0.0-rc2 executes the upstream AO body at widths 8 and 16 (upstream default: 64). '
        'The four arms issue 217,070 and 915,981 GC slots respectively. (a) Peak GC groups remain '
        '6/6/6/9 over the two measured sizes. (b) Selected GC bytes relative to each arm’s own '
        'spatial control, at the peak and after rendering. The counts include per-group metadata '
        'and observers; PoisonCap also includes mapping rounding. Sublet retains more groups '
        'than its control after AO 16. Baseline layouts differ, so these ratios do not rank full '
        'adaptation cost. Three repetitions coincide. Nodes and revocation storage are excluded.'),
    '04-ffmpeg-snapshots': (
        'FFmpeg 9.0.1: selected PoisonCap adapter snapshot backing in two adapted FATE inputs '
        'and repeated 30-frame streams. Bars show temporal-mode high-water bytes; spatial-mode '
        'peaks and final temporal-mode bytes are zero. Three processes per cell coincide. The '
        'snapshot peak stays at 35.4 KiB across 1, 4 and 16 repeated streams; it depends on the '
        'input and does not grow with completed frames in this range. All four application arms '
        'have identical reuse bins for each input. This component is neither total memory nor '
        'a universal PoisonCap lower bound; adapted FATE inputs are not official FATE scores.'),
    '05-all-workloads-reuse': (
        'All eight admitted workloads, using the same four styles, axes, bin edges and all-issues '
        'denominator as Figure 1. Every curve represents three complete application processes '
        'with identical histograms (96 processes total). Work sizes and allocator boundaries '
        'remain separate. The main-panel selection is exploratory, not preregistered. '
        'The artifact retains counts, paired differences, input hashes, source-campaign methods '
        'and successful raw-log revalidation. A plateau is the observed reuse fraction, not '
        'an estimate of eventual reuse of every released object.'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def cdf(bins, issues):
    require(len(bins) == 32 and all(type(n) is int and n >= 0 for n in bins),
            'expected 32 nonnegative integer histogram counts')
    require(type(issues) is int and issues > 0 and sum(bins) <= issues,
            'reuse denominator must include all successful issues')
    # Bin 31 saturates in some observers. Do not invent a finite upper bound.
    require(bins[-1] == 0, 'overflow bin needs separate censored-tail reporting')
    return np.cumsum(bins) * 100.0 / issues


def checked_cell(rows):
    require(len(rows) == 12 and {(r['arm'], r['rep']) for r in rows} ==
            {(a, rep) for a in ARMS for rep in range(3)},
            'expected exactly three processes for each of four arms')
    require(len({r['issues'] for r in rows}) == 1, 'allocation demand differs')
    cell = {}
    for arm in ARMS:
        samples = sorted((r for r in rows if r['arm'] == arm), key=lambda r: r['rep'])
        for r in samples:
            cdf(r['bins'], r['issues'])
            require(sum(r['bins']) == r['reuses'], 'histogram does not reconcile')
        require(all(r['bins'] == samples[0]['bins'] for r in samples),
                'repetitions differ: plot their range instead of one curve')
        cell[arm] = samples[0]
    return cell


class Inputs:
    def __init__(self):
        self.hashes = {}

    def track(self, path):
        self.hashes[str(path.relative_to(HERE))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(self, name):
        return json.loads(self.track(ROOT / name).read_text())


def load_reuse(inputs):
    groups = {}
    sqlite = inputs.read('sqlite-reuse-gaps-20260927/runs.json')
    groups['sqlite-main'] = [dict(arm=ARMS[0] if r['arm'] == 'capstone' else r['arm'],
        rep=r['rep']-1, issues=r['allocations'], reuses=r['reuses'], bins=r['bins']) for r in sqlite]
    mruby = {}
    for width, prefix in ((8, 'ao8/'), (16, '')):
        data = inputs.read(f'mruby-gc-memory-20260927/{prefix}summary.json')
        require(data['benchmark'] == f'mruby 4.0.0-rc2 bm_ao_render.rb width={width}',
                'unexpected mruby source/workload')
        runs = data['runs']
        oracle = {8: '28060a9790593ace825b7552ab5d1878cf3e95be3be08ee738f28cde4cee5d11',
                  16: 'cfd73bb1de97514b2adacbda3a377e5172aecbe48819f654071d1996359ca634'}[width]
        for r in runs:
            require(r['stdout_sha256'] == oracle, 'mruby output oracle differs')
            phases = r['phases']
            require([p['phase'] for p in phases] == ['startup', 'before', 'after', 'exit'],
                    'incomplete mruby phases')
            p = phases[2]
            require(phases[-1]['pages'] == 0 and phases[-1]['issues'] == p['issues'],
                    'mruby exit or work accounting differs')
            page_bytes = (p['page_payload_bytes'] + p['page_metadata_bytes']
                          if r['platform'] == 'capstone' else p['mapped_page_bytes'])
            require(r['selected_peak_bytes'] == p['peak_pages'] * page_bytes and
                    r['selected_after_bytes'] == p['pages'] * page_bytes and
                    r['gap_le_1023'] == sum(r['bins'][:10]), 'mruby byte accounting differs')
        groups[f'mruby-ao{width}'] = [dict(arm=r['arm'], rep=r['rep'],
            issues=r['phases'][-1]['issues'], reuses=r['phases'][-1]['reissues'], bins=r['bins'])
            for r in runs]
        mruby[width] = runs
    for arm in ARMS:
        require(len({r['binary_sha256'] for runs in mruby.values() for r in runs if r['arm'] == arm}) == 1,
                'mruby binary changed with work size')
    require(len({r['workload_sha256'] for runs in mruby.values() for r in runs}) == 1,
            'AO workload source changed')
    fate = inputs.read('ffmpeg-fate-four-arm-20260928/summary.json')
    require(fate['valid_attempts'] == len(fate['runs']) == 24, 'incomplete FATE matrix')
    mapping = {'capstone-pool0': ARMS[0], 'capstone-pool2': ARMS[1],
               ARMS[2]: ARMS[2], ARMS[3]: ARMS[3]}
    for case, key, frames, oracle in (
        ('xvid_vlc_trac7411', 'ffmpeg-xvid', 20, '45c371036640c8ae77bb5efdf4447b1625388fc8b4d5135e0a76d532bcd04cf2'),
        ('resize_down-up', 'ffmpeg-resize', 150, '939f2e7a46ace2030fc688ff809d666473d17ca5944e442d0722c6f02ce7e14c')):
        runs = [r for r in fate['runs'] if r['case'] == case]
        require(all(r['frames'] == frames and r['stdout_sha256'] == oracle for r in runs),
                'FATE output or work differs')
        groups[key] = [dict(arm=mapping[r['arm']], rep=r['repetition'],
            issues=r['issues'], reuses=r['reuses'], bins=r['bins']) for r in runs]
    streams = inputs.read('ffmpeg-reuse-gaps-20260927/summary.json')['runs']
    require(len(streams) == 36, 'incomplete repeated-stream matrix')
    mapping = dict(zip(('Capstone original', 'Capstone + Sublet',
                       'PoisonCap spatial', 'PoisonCap temporal'), ARMS))
    for n in (1, 4, 16):
        runs = [r for r in streams if r['batches'] == n]
        require(all(r['frame_count'] == 30*n for r in runs) and
                len({r['frame_oracle_sha256'] for r in runs}) == 1, 'FFmpeg oracle/work differs')
        groups[f'ffmpeg-streams{n}'] = [dict(arm=mapping[r['arm']], rep=r['repetition'],
            issues=r['issues'], reuses=r['reuses'], bins=r['bins']) for r in runs]
    return {key: checked_cell(rows) for key, rows in groups.items()}, mruby, fate['runs'], streams


def load_sqlite_memory(inputs):
    folder = ROOT / 'sqlite-normalized-memory-20260927'
    parser = module('plot-sqlite-normalized')
    inputs.track(HERE / 'plot-sqlite-normalized.py')
    runs = inputs.read('sqlite-normalized-memory-20260927/runs.json')['runs']
    reference = inputs.read('sqlite-normalized-memory-20260927/oracles.json')
    result = {a: {} for a in ARMS}
    for run in runs:
        if run['profile'] != 'churn':
            continue
        arm = ARMS[0] if run['arm'] == 'capstone' else run['arm']
        path = inputs.track(folder / run['stdout'])
        raw = parser.readtext(path)
        require(hashlib.sha256(raw.encode()).hexdigest() == run['stdout_sha256'],
                'SQLite raw transcript hash differs')
        rows, begins, ends, oracles = parser.parse(raw)
        require(run['status'] == 'completed' and ends == {i: 0 for i in range(17)} and
                'STUDY-COMPLETE units=17' in raw and all(v == 1 for v in begins.values()),
                'incomplete or changed SQLite work')
        require(all(oracles[u] == reference[str(begins[u])] for u in ends), 'SQLite output oracle differs')
        if arm.startswith('capstone'):
            require('DROPPED 0 RC 0' in raw, 'Capstone completion gate failed')
        units = [r for r in rows if r['phase'] == -1]
        require(len(units) == 17 and [r['unit'] for r in units] == list(range(17)) and
                all(r['live'] == r['oom'] == 0 for r in units), 'incomplete SQLite ledgers')
        rep = run['rep']-1
        require(rep not in result[arm], 'duplicate SQLite process')
        result[arm][rep] = units
    for arm, runs in result.items():
        require(set(runs) == {0, 1, 2}, 'incomplete SQLite memory matrix')
        require(all(runs[r] == runs[0] for r in (1, 2)), 'SQLite memory variation requires ranges')
    return result


def line_style(index):
    return dict(color=COLORS[index], linestyle='--' if index % 2 == 0 else '-',
                linewidth=1.1 if index % 2 == 0 else .85,
                marker=MARKERS[index], markersize=3.3, markeredgewidth=.8,
                markerfacecolor='white')


def legend(fig, four=True):
    indices = range(4) if four else (1, 3)
    handles = [Line2D([], [], label=LABELS[i], **line_style(i)) for i in indices]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, 1.0),
               ncol=len(handles), frameon=False, handlelength=2.4, columnspacing=1.4)


def clean_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='.87', linewidth=.45)
    ax.tick_params(length=2.5, pad=2)


def reuse_axis(ax):
    ax.set_xscale('log', base=2)
    ax.set_xlim(1, 2**20)
    ax.set_xticks([1, 16, 256, 4096, 65536, 2**20], ['1', '16', '256', '4K', '64K', '1M'])
    clean_axis(ax)


def draw_cdf(ax, cell):
    for i, arm in enumerate(ARMS):
        r = cell[arm]
        # Stagger symbols among *observed* bucket edges, never jitter the data.
        ax.step(EDGES, cdf(r['bins'], r['issues']), where='post',
                markevery=(i, 4), **line_style(i))
    reuse_axis(ax)
    ax.set_ylim(0, 103)
    ax.set_yticks([0, 25, 50, 75, 100])


def reuse_figure(data):
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.3), sharex=True, sharey='row',
                             gridspec_kw={'height_ratios': [1.5, 1]})
    fig.subplots_adjust(left=.085, right=.985, top=.81, bottom=.115, hspace=.19, wspace=.19)
    legend(fig)
    for j, key in enumerate(MAIN):
        cell = data[key]
        ax = axes[0, j]
        draw_cdf(ax, cell)
        ax.set_title(f'({chr(97+j)}) {TITLES[key]}', pad=7)
        ax.text(.96, .06, f"{cell[ARMS[0]]['issues']:,} issues", transform=ax.transAxes,
                ha='right', fontsize=7.5)
        low = axes[1, j]
        for original, protected in PAIRS:
            i = ARMS.index(protected)
            loss = cdf(cell[original]['bins'], cell[original]['issues']) - cdf(cell[protected]['bins'], cell[protected]['issues'])
            low.step(EDGES, loss, where='post', markevery=(i, 4), **line_style(i))
            if loss.max() > 10:
                low.text(.98, .9, f'max. {loss.max():.2f} pp', transform=low.transAxes,
                         ha='right', va='top', color=COLORS[i], fontsize=8)
        reuse_axis(low)
        low.axhline(0, color='.6', linewidth=.45, zorder=0)
        low.set_ylim(-6, 103)
        low.set_yticks([0, 50, 100])
        low.set_xlabel('Release-to-reissue gap (issues)')
    axes[0, 0].set_ylabel('Reissues / all issues (%)')
    axes[1, 0].set_ylabel('Control − protected (pp)')
    axes[1, 2].text(.5, .55, 'All four curves coincide', transform=axes[1, 2].transAxes,
                    ha='center', fontsize=8)
    return fig


def supplement(data):
    fig, axes = plt.subplots(4, 2, figsize=(7.05, 8.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.985, top=.92, bottom=.065, hspace=.58, wspace=.18)
    legend(fig)
    order = ('sqlite-main', 'mruby-ao8', 'mruby-ao16', 'ffmpeg-xvid', 'ffmpeg-resize',
             'ffmpeg-streams1', 'ffmpeg-streams4', 'ffmpeg-streams16')
    for i, (ax, key) in enumerate(zip(axes.flat, order)):
        draw_cdf(ax, data[key])
        ax.set_title(f'({chr(97+i)}) {TITLES[key]}', pad=6)
        if i % 2 == 0:
            ax.set_ylabel('Reissues / all issues (%)')
        ax.tick_params(labelbottom=True)
    for ax in axes[-1]:
        ax.set_xlabel('Release-to-reissue gap (issues)')
    return fig


def sqlite_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.7))
    fig.subplots_adjust(left=.095, right=.97, top=.76, bottom=.21, wspace=.32)
    legend(fig, four=False)
    for ax, metric, title in zip(axes, ('ever', 'peak'),
                                 ('(a) Allocated address coverage', '(b) Peak selected allocator bytes')):
        for base, protected in PAIRS:
            def values(arm):
                return np.array([r['ever'] if metric == 'ever' else r['peak_held'] + r['metadata']
                                 for r in data[arm][0]], dtype=float)
            ratio = values(protected)/values(base)
            i = ARMS.index(protected)
            ax.plot(range(17), ratio, markevery=(i-1, 4), **line_style(i))
            at = 16 if metric == 'ever' else int(np.argmax(ratio[1:]))+1
            y = ratio[at]
            label = f'{y:.2f}×' if metric == 'ever' else f'peak {y:.2f}×'
            # Put a within-run maximum at its measured unit, not at the endpoint.
            if metric == 'peak':
                ax.plot(at, y, linestyle='none', marker=MARKERS[i], color=COLORS[i],
                        markerfacecolor='white', markersize=3.3)
            ax.annotate(label, (at, y), xytext=(0, 6 if i == 1 else -13),
                        textcoords='offset points', ha='left' if at < 4 else 'right', color=COLORS[i])
        ax.axhline(1, color='.5', linewidth=.6, linestyle=':')
        ax.axvspan(-.5, .5, color='.93', zorder=-1)
        ax.set(xlim=(-.5, 16.5), ylim=(0, 5.35), xticks=[0, 4, 8, 12, 16],
               yticks=[0, 1, 2, 3, 4, 5], title=title, xlabel='Completed unit (0 = warmup)',
               ylabel='Protected / own original (×)')
        clean_axis(ax)
    return fig


def mruby_figure(runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.95))
    fig.subplots_adjust(left=.075, right=.98, top=.76, bottom=.22, wspace=.36)
    samples = {w: {a: next(r for r in rs if r['arm'] == a) for a in ARMS} for w, rs in runs.items()}
    for w, rs in runs.items():
        for arm in ARMS:
            ref = samples[w][arm]
            require(all((r['selected_peak_bytes'], r['selected_after_bytes'], r['phases']) ==
                        (ref['selected_peak_bytes'], ref['selected_after_bytes'], ref['phases'])
                        for r in rs if r['arm'] == arm), 'mruby memory variation requires ranges')
    handles = []
    for i, arm in enumerate(ARMS):
        fc = 'white' if i % 2 == 0 else COLORS[i]
        axes[0].bar(np.arange(2)+(i-1.5)*.19,
                    [samples[w][arm]['phases'][2]['peak_pages'] for w in (8, 16)],
                    width=.17, facecolor=fc, edgecolor=COLORS[i], linewidth=.8,
                    hatch='///' if i % 2 == 0 else None)
        handles.append(Patch(facecolor=fc, edgecolor=COLORS[i],
                             hatch='///' if i % 2 == 0 else None, label=LABELS[i]))
    axes[0].set(xticks=[0, 1], xticklabels=['AO 8\n217,070 issues', 'AO 16\n915,981 issues'],
                ylim=(0, 10.5), yticks=[0, 3, 6, 9], ylabel='Peak GC groups (1,024 slots each)',
                title='(a) More work, same peak group count')
    for base, protected in PAIRS:
        i = ARMS.index(protected)
        ratios = [samples[w][protected][field]/samples[w][base][field]
                  for w in (8, 16) for field in ('selected_peak_bytes', 'selected_after_bytes')]
        xs = np.arange(4) + (-.12 if i == 1 else .12)
        axes[1].plot(xs, ratios, linestyle='none', marker=MARKERS[i], color=COLORS[i],
                     markersize=4, markerfacecolor=COLORS[i])
        for x, value in zip(xs, ratios):
            axes[1].annotate(f'{value:.2f}', (x, value), xytext=(-3 if i == 1 else 3, -12 if i == 1 else 6),
                             textcoords='offset points', ha='center', color=COLORS[i], fontsize=7)
    axes[1].axhline(1, color='.5', linewidth=.7, linestyle=':')
    axes[1].axvline(1.5, color='.85', linewidth=.6)
    axes[1].set(xlim=(-.5, 3.5), xticks=range(4), xticklabels=['Peak\nAO 8', 'After\nAO 8', 'Peak\nAO 16', 'After\nAO 16'],
                ylim=(0, 2.5), yticks=[0, .5, 1, 1.5, 2, 2.5],
                ylabel='Selected bytes / own spatial (×)', title='(b) Metadata and retained groups count')
    for ax in axes:
        clean_axis(ax)
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(.53, 1.0), columnspacing=1.4)
    return fig


def ffmpeg_figure(fate, streams):
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75))
    fig.subplots_adjust(left=.09, right=.98, top=.77, bottom=.22, wspace=.36)
    metrics = []
    for rows, key, cases, name in (
        (fate, 'case', ('xvid_vlc_trac7411', 'resize_down-up'), 'adapter'),
        (streams, 'batches', (1, 4, 16), 'rewrite')):
        values = []
        for case in cases:
            temporal = [r[name] for r in rows if r[key] == case and r['arm'].lower() == 'poisoncap-temporal']
            # The historical stream summary has presentation labels with spaces.
            if not temporal:
                temporal = [r[name] for r in rows if r[key] == case and r['arm'] == 'PoisonCap temporal']
            spatial = [r[name] for r in rows if r[key] == case and r['arm'] in ('poisoncap-spatial', 'PoisonCap spatial')]
            require(len(temporal) == len(spatial) == 3 and all(v == temporal[0] for v in temporal),
                    'FFmpeg adapter repetitions differ')
            require(all(v['snapshot_peak'] == v['snapshot_bytes'] == 0 for v in spatial),
                    'FFmpeg spatial snapshot counter differs')
            require(temporal[0]['snapshot_bytes'] == 0, 'FFmpeg final snapshot backing retained')
            values.append(temporal[0]['snapshot_peak']/1024)
        metrics.append(values)
    for j, (ax, values) in enumerate(zip(axes, metrics)):
        x = np.arange(len(values))
        ax.bar(x, values, width=.5, color=ORANGE, alpha=.85)
        ax.plot(x-.08, np.zeros(len(x)), linestyle='none', marker='+', color=ORANGE,
                markersize=6, clip_on=False)
        ax.plot(x+.08, np.zeros(len(x)), linestyle='none', marker='s', color=ORANGE,
                markerfacecolor='white', markersize=4, clip_on=False)
        for at, value in zip(x, values):
            ax.text(at, value+7, f'{value:.1f}', ha='center', fontsize=8)
        ax.set(ylim=(0, 260), yticks=[0, 64, 128, 192, 256], ylabel='Snapshot backing (KiB)')
        clean_axis(ax)
    axes[0].set(xticks=[0, 1], xticklabels=['Xvid\n20 frames', 'Resize\n150 frames'],
                title='(a) Adapted FATE inputs')
    axes[1].set(xticks=[0, 1, 2], xticklabels=['1 stream\n30 frames', '4 streams\n120 frames', '16 streams\n480 frames'],
                title='(b) Repeated 30-frame input')
    fig.legend(handles=[Patch(facecolor=ORANGE, label='Temporal: peak'),
        Line2D([], [], color=ORANGE, marker='+', ls='', label='Spatial: peak'),
        Line2D([], [], color=ORANGE, marker='s', mfc='white', ls='', label='Temporal: after decode')],
        loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.53, 1.0))
    return fig


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def export_data(out, data, sqlite, mruby, fate, streams):
    bins, paired, counts = [], [], []
    for key, cell in data.items():
        for arm, r in cell.items():
            counts.append(dict(workload=key, arm=arm, processes=3, issues=r['issues'], reissues=r['reuses'],
                               reissue_percent=100*r['reuses']/r['issues']))
            bins.extend(dict(workload=key, arm=arm, gap_upper=int(edge), count=count, cumulative_percent=float(y))
                        for edge, count, y in zip(EDGES, r['bins'], cdf(r['bins'], r['issues'])))
        for base, protected in PAIRS:
            delta = cdf(cell[base]['bins'], cell[base]['issues']) - cdf(cell[protected]['bins'], cell[protected]['issues'])
            at = int(np.argmax(delta))
            paired.append(dict(workload=key, protected=protected, processes_per_arm=3,
                max_reuse_deficit_pp=max(0.0, float(delta[at])),
                first_max_gap_upper=int(EDGES[at]) if delta[at] > 0 else None,
                min_signed_deficit_pp=float(min(delta)),
                endpoint_deficit_pp=float(delta[-1])))
    write_csv(out / 'reuse-bins.csv', bins)
    write_csv(out / 'reuse-counts.csv', counts)
    write_csv(out / 'paired-reuse.csv', paired)
    memory = []
    for arm, reps in sqlite.items():
        for rep, rows in reps.items():
            for r in rows:
                memory.append(dict(application='sqlite', arm=arm, repetition=rep, unit=r['unit'],
                    address_coverage_bytes=r['ever'], selected_peak_bytes=r['peak_held']+r['metadata'],
                    selected_after_bytes=r['held']+r['metadata'], metadata_bytes=r['metadata']))
    write_csv(out / 'sqlite-memory.csv', memory)
    memory = []
    for width, rows in mruby.items():
        for r in rows:
            p = r['phases'][2]
            memory.append(dict(application='mruby', width=width, arm=r['arm'], repetition=r['rep'],
                peak_groups=p['peak_pages'], after_groups=p['pages'],
                selected_peak_bytes=r['selected_peak_bytes'], selected_after_bytes=r['selected_after_bytes'],
                observer_bytes_per_group=p['observer_bytes']))
    write_csv(out / 'mruby-memory.csv', memory)
    memory = []
    for rows, case, adapter in ((fate, 'case', 'adapter'), (streams, 'batches', 'rewrite')):
        for r in rows:
            if r[adapter] is not None:
                memory.append(dict(application='ffmpeg', workload=r[case], arm=r['arm'], repetition=r['repetition'],
                    snapshot_peak_bytes=r[adapter]['snapshot_peak'], snapshot_after_bytes=r[adapter]['snapshot_bytes']))
    write_csv(out / 'ffmpeg-memory.csv', memory)


def verify_raw(inputs, out):
    """Re-run the original campaign validators without booting any guests.

    Existing collectors use their recorded /tmp/capstone build paths. The
    FATE archive has a portable layout and is restored in an isolated folder.
    """
    scratch = Path('/tmp/capstone')
    archives = {}
    for folder in ('sqlite-reuse-gaps-20260927', 'mruby-gc-memory-20260927',
                   'ffmpeg-reuse-gaps-20260927', 'ffmpeg-fate-four-arm-20260928'):
        pin = inputs.read(folder + '/archive.json')
        path = Path(pin.get('archive', pin.get('path')))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == pin['sha256'], 'raw archive hash differs: '+str(path))
        archives[folder] = dict(path=str(path), sha256=digest)
    checks = []
    with tempfile.TemporaryDirectory(prefix='paper-validate-', dir=scratch) as temp:
        temp = Path(temp)
        inputs.track(HERE / 'plot-sqlite-reuse-gaps.py')
        subprocess.run([sys.executable, str(HERE / 'plot-sqlite-reuse-gaps.py'),
                        str(scratch / 'sqlite-reuse-gap-20260927'), str(temp / 'sqlite')],
                       check=True, capture_output=True, text=True)
        require(json.loads((temp / 'sqlite/runs.json').read_text()) ==
                inputs.read('sqlite-reuse-gaps-20260927/runs.json'), 'SQLite raw-derived data differ')
        checks.append(dict(campaign='sqlite-reuse', processes=12, exact_records_match=True))
        m = module('plot-mruby-gc-memory')
        inputs.track(HERE / 'plot-mruby-gc-memory.py')
        for width, capstone, poisoncap, prefix in (
            (8, 'mruby-gc-ao8-capstone-run', 'mruby-gc-ao8-poisoncap-run', 'ao8/'),
            (16, 'mruby-gc-v3-ao-run', 'mruby-poisoncap-ao-campaign', '')):
            rows = m.collect(scratch / capstone, scratch / poisoncap, width)
            require(rows == inputs.read(f'mruby-gc-memory-20260927/{prefix}summary.json')['runs'],
                    'mruby raw-derived data differ')
            checks.append(dict(campaign=f'mruby-ao{width}', processes=12, exact_records_match=True))
        f = module('plot-ffmpeg-reuse-gaps')
        inputs.track(HERE / 'plot-ffmpeg-reuse-gaps.py')
        rows = f.collect(scratch / 'ffmpeg-reuse-capstone-sdk-runs', scratch / 'ffmpeg-reuse-poisoncap-runs')
        require(rows == inputs.read('ffmpeg-reuse-gaps-20260927/summary.json')['runs'],
                'FFmpeg repeated-stream raw-derived data differ')
        checks.append(dict(campaign='ffmpeg-streams', processes=36, exact_records_match=True))
        restored = temp / 'fate'
        with tarfile.open(archives['ffmpeg-fate-four-arm-20260928']['path']) as archive:
            archive.extractall(restored, filter='data')
        f = module('collect-ffmpeg-fate')
        inputs.track(HERE / 'collect-ffmpeg-fate.py')
        data = f.collect(restored / 'capstone-runs', restored / 'cheribsd-both',
                         scratch / 'poisoncap-l2-probe/rootfs/boot/kernel/kernel')
        require(data == inputs.read('ffmpeg-fate-four-arm-20260928/summary.json'),
                'FFmpeg FATE raw-derived data differ')
        checks.append(dict(campaign='ffmpeg-fate', processes=24, exact_records_match=True))
    require(sum(row['processes'] for row in checks) == 96, 'incomplete raw audit')
    report = dict(schema=1, new_application_runs=False, raw_archives=archives, checks=checks,
                  sqlite_memory_processes_revalidated_from_committed_raw=12,
                  input_sha256=dict(inputs.hashes))
    (out / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-raw', action='store_true',
                        help='also rerun original validators on restored /tmp/capstone archives')
    args = parser.parse_args()
    inputs = Inputs()
    inputs.track(Path(__file__).resolve())
    data, mruby, fate, streams = load_reuse(inputs)
    sqlite = load_sqlite_memory(inputs)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.verify_raw:
        verify_raw(inputs, args.out)
    export_data(args.out, data, sqlite, mruby, fate, streams)
    plt.rcParams.update({'font.family': 'STIXGeneral', 'font.size': 8.5,
        'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    figures = [('01-reuse-and-control', lambda: reuse_figure(data)),
               ('02-sqlite-memory', lambda: sqlite_figure(sqlite)),
               ('03-mruby-memory', lambda: mruby_figure(mruby)),
               ('04-ffmpeg-snapshots', lambda: ffmpeg_figure(fate, streams)),
               ('05-all-workloads-reuse', lambda: supplement(data))]
    metadata = {'CreationDate': None, 'ModDate': None, 'Title': 'Measured application memory behavior'}
    with PdfPages(args.out / 'application-memory-figures.pdf', metadata=metadata) as book:
        for index, (name, draw) in enumerate(figures, 1):
            fig = draw()
            fig.savefig(args.out / (name+'.pdf'), metadata=metadata)
            fig.savefig(args.out / (name+'.png'), dpi=240)
            # The review packet is self-contained; standalone figures stay at
            # exactly 7.05 inches and leave caption typesetting to the paper.
            caption = textwrap.fill(f'Figure {index}. '+CAPTIONS[name], 112)
            extra = .25 + len(caption.splitlines())*.15
            width, height = fig.get_size_inches()
            positions = [ax.get_position().bounds for ax in fig.axes]
            fig.set_size_inches(width, height+extra)
            for ax, (left, bottom, w, h) in zip(fig.axes, positions):
                ax.set_position([left, (bottom*height+extra)/(height+extra), w, h*height/(height+extra)])
            fig.text(.075, (extra-.1)/(height+extra), caption, fontsize=9, va='top', linespacing=1.2)
            book.savefig(fig)
            plt.close(fig)
    snippets = ['% Use with graphicx; set this directory before including this file.',
                r'\providecommand{\appMemoryFigureRoot}{.}']
    for name, _ in figures:
        caption = CAPTIONS[name].replace('’', "'")
        caption = caption.replace('Figure 5', r'Figure~\ref{fig:app-memory-05-all-workloads-reuse}')
        caption = caption.replace('Figure 1', r'Figure~\ref{fig:app-memory-01-reuse-and-control}')
        snippets += [r'\begin{figure*}[t]', r'  \centering',
            r'  \includegraphics[width=\textwidth]{\appMemoryFigureRoot/'+name+'.pdf}',
            '  \\caption{'+caption+'}', '  \\label{fig:app-memory-'+name+'}', r'\end{figure*}', '']
    (args.out / 'figures.tex').write_text('\n'.join(snippets).rstrip()+'\n')
    manifest = dict(schema=1, new_application_runs=False, reuse_processes=96, sqlite_memory_processes=12,
        repetitions_per_cell=3, repetitions_coincide=True, width_inches=7.05,
        plotted_denominator='all successful inner-boundary issues; immediate gap = 1',
        main_workloads=list(MAIN), all_workloads=list(data),
        main_selection='SQLite available matched workload; larger measured mruby size; longer adapted FATE input. Exploratory, not preregistered.',
        inputs_sha256=inputs.hashes,
        rendering_versions={'matplotlib': matplotlib.__version__, 'numpy': np.__version__},
        accounting='selected allocator / GC-group / adapter storage; no total-memory or physical-working-set inference')
    (args.out / 'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Validated 96 reuse processes and 12 SQLite memory processes; wrote {len(figures)} figures to {args.out}')


if __name__ == '__main__':
    main()
