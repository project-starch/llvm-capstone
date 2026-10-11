#!/usr/bin/env python3
"""Validate complete application runs and plot same-start release gaps.

Input is the output of actual speedtest1 processes, never an allocation
request replay. The denominator is all successful memsys5 allocations.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import statistics
from pathlib import Path

ARMS = ('capstone', 'capstone-sublet', 'poisoncap-spatial', 'poisoncap-temporal')
PAIRS = (('capstone', 'capstone-sublet'), ('poisoncap-spatial', 'poisoncap-temporal'))
COLOR = {'capstone-sublet': '#0072B2', 'poisoncap-temporal': '#D55E00'}
TOTAL = re.compile(r'STUDY-GAP-TOTAL unit=(\d+) allocs=(\d+) reuses=(\d+)')
BUCKET = re.compile(r'STUDY-GAP unit=(\d+) pair=(\d+) a=(\d+) b=(\d+)')
LEDGER = re.compile(r'STUDY-LEDGER unit=(\d+) phase=-1 [^\n]*?allocs=(\d+) reused_starts=(\d+) [^\n]*?observer=(\d+)')
ORACLE = re.compile(r'STUDY-ORACLE phase=(\d+) rows=(\d+) hash=([0-9a-f]+)')


def parse(raw, expected):
    totals = {}
    buckets = {}
    for m in TOTAL.finditer(raw):
        unit, allocations, reuses = map(int, m.groups())
        if unit in totals:
            raise ValueError(f'duplicate gap total for unit {unit}')
        totals[unit] = (allocations, reuses)
    for m in BUCKET.finditer(raw):
        unit, pair, a, b = map(int, m.groups())
        if pair not in range(16) or (unit, pair) in buckets:
            raise ValueError(f'duplicate or invalid gap pair {unit},{pair}')
        buckets[unit, pair] = (a, b)
    ends = {}
    for m in LEDGER.finditer(raw):
        unit, allocs, starts, observer = map(int, m.groups())
        if unit in ends:
            raise ValueError(f'duplicate unit ledger {unit}')
        ends[unit] = (allocs, starts, observer)
    if set(totals) != set(range(expected)) or set(ends) != set(totals):
        raise ValueError('incomplete gap totals or unit ledgers')
    if set(buckets) != {(u, p) for u in totals for p in range(16)}:
        raise ValueError('incomplete gap bins')
    previous_bins = [0]*32
    previous_allocs = previous_reuses = 0
    prefix = totals[0][0] - ends[0][0]
    if prefix < 0 or prefix > 1000:
        raise ValueError('unexpected setup allocation prefix')
    previous_allocs = prefix
    all_reused_starts = 0
    for unit in range(expected):
        allocations, reuses = totals[unit]
        all_reused_starts += ends[unit][1]
        b = [n for p in range(16) for n in buckets[unit, p]]
        if (allocations != previous_allocs + ends[unit][0] or
            reuses != sum(b) or reuses > allocations or
            reuses < previous_reuses or any(x < y for x, y in zip(b, previous_bins)) or
            ends[unit][2] < 500_000):
            raise ValueError(f'inconsistent cumulative gaps at unit {unit}')
        previous_allocs, previous_reuses, previous_bins = allocations, reuses, b
    if 'STUDY-COMPLETE units='+str(expected) not in raw:
        raise ValueError('application process did not complete')
    if previous_reuses != all_reused_starts:
        raise ValueError('gap counts differ from the independent same-start reuse counter')
    return {'allocations': previous_allocs-prefix, 'setup_allocations': prefix,
            'reuses': previous_reuses, 'bins': previous_bins,
            'observer_bytes': ends[expected-1][2]}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('scratch', type=Path)
    ap.add_argument('result', type=Path)
    args = ap.parse_args()
    scratch, result = args.scratch, args.result
    reference = json.loads((Path(__file__).parent/'results/sqlite-normalized-memory-20260927/oracles.json').read_text())['1']
    old_root = Path(__file__).parent/'results/sqlite-normalized-memory-20260927'
    old_phase = {}
    with (old_root/'phase-ledger.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            if row['profile']=='churn' and row['complete']=='True':
                key=(row['arm'],int(row['rep']),int(row['unit']),int(row['phase']))
                if key in old_phase:raise ValueError(f'duplicate old observation {key}')
                old_phase[key]=row
    spec=importlib.util.spec_from_file_location('sqlite_memory_plot',
              Path(__file__).with_name('plot-sqlite-normalized.py'))
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    rows = []
    manifests = ((scratch/'nodes4m/capstone-runs.jsonl', scratch/'nodes4m'),
                 (scratch/'cheri-runs.jsonl', scratch))
    for manifest, directory in manifests:
        for run in load_jsonl(manifest):
            if run['profile'] != 'churn':
                continue
            arm, rep = run['arm'], run['rep']
            if arm not in ARMS or rep not in (1, 2, 3) or run['status'] != 'completed':
                raise ValueError(f'incomplete requested cell: {arm} {rep}')
            path = directory/f'{arm}-churn-{rep}.stdout'
            raw = path.read_text(errors='replace')
            if [list(m) for m in ORACLE.findall(raw)] != reference*17:
                raise ValueError(f'SQL result oracle mismatch: {arm} {rep}')
            samples, _, _, _ = prior.parse(raw)
            if len(samples)!=len([k for k in old_phase if k[:2]==(arm,rep)]):
                raise ValueError(f'allocator phase count changed: {arm} {rep}')
            for sample in samples:
                key=(arm,rep,sample['unit'],sample['phase'])
                old=old_phase[key]
                if any(value!=int(old[field]) for field,value in sample.items() if field!='observer'):
                    raise ValueError(f'allocator ledger changed after gap observation: {key}')
            parsed = parse(raw, 17)
            if run['binary_sha256'] != hashlib.sha256((scratch/(f'{arm}-build/sqlite_silicon.dom' if arm.startswith('capstone') else arm)).read_bytes()).hexdigest():
                raise ValueError(f'binary identity mismatch: {arm} {rep}')
            row = {'arm': arm, 'rep': rep, 'stdout': str(path.relative_to(scratch)),
                   'stdout_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   **parsed}
            rows.append(row)
    if {(r['arm'], r['rep']) for r in rows} != {(arm, rep) for arm in ARMS for rep in (1, 2, 3)}:
        raise ValueError('12 complete four-arm repetitions required')

    result.mkdir(parents=True, exist_ok=True)
    (result/'runs.json').write_text(json.dumps(rows, indent=2)+'\n')
    measures={}
    for arm in ARMS:
        runs=[r for r in rows if r['arm']==arm]
        values={
            'reuse_fraction': [r['reuses']/r['allocations'] for r in runs],
            'immediate_fraction': [r['bins'][0]/r['allocations'] for r in runs],
            'within_15_fraction': [sum(r['bins'][:4])/r['allocations'] for r in runs],
            'within_4095_fraction': [sum(r['bins'][:12])/r['allocations'] for r in runs],
        }
        measures[arm]={k:{'median':statistics.median(v),'min':min(v),'max':max(v)}
                       for k,v in values.items()}
        measures[arm].update(allocations=sorted({r['allocations'] for r in runs}),
                             reuses=sorted({r['reuses'] for r in runs}),
                             observer_bytes=sorted({r['observer_bytes'] for r in runs}),
                             setup_allocations=sorted({r['setup_allocations'] for r in runs}))
    comparisons={}
    for base, protected in PAIRS:
        paired=[]
        for rep in (1,2,3):
            b=next(r for r in rows if r['arm']==base and r['rep']==rep)
            p=next(r for r in rows if r['arm']==protected and r['rep']==rep)
            if b['allocations']!=p['allocations']:
                raise ValueError(f'paired application allocation demand changed: {base}/{protected} rep {rep}')
            paired.append({'rep':rep,
                           'reuse_percentage_points':100*(p['reuses']-b['reuses'])/b['allocations'],
                           'immediate_percentage_points':100*(p['bins'][0]-b['bins'][0])/b['allocations'],
                           'within_15_percentage_points':100*(sum(p['bins'][:4])-sum(b['bins'][:4]))/b['allocations']})
        comparisons[protected]=paired
    (result/'summary.json').write_text(json.dumps({'arms':measures,'paired_effects':comparisons},indent=2)+'\n')
    with (result/'bins.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['arm','rep','gap_lower_allocations','gap_upper_allocations','reuse_count','all_allocations','cumulative_fraction'])
        for r in rows:
            total = 0
            for b, count in enumerate(r['bins']):
                total += count
                writer.writerow([r['arm'], r['rep'], 1<<b, (1<<(b+1))-1,
                                 count, r['allocations'], total/r['allocations']])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
                         'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.45))
    fig.subplots_adjust(left=.10, right=.98, bottom=.22, top=.76, wspace=.25)
    handles=[]
    for base, prot in PAIRS:
        handles += [Line2D([],[],color=COLOR[prot],ls='--',lw=2.2,
                           label='Capstone original' if base=='capstone' else 'CheriBSD original'),
                    Line2D([],[],color=COLOR[prot],ls='-',lw=1.2,
                           label='Sublet' if prot=='capstone-sublet' else 'PoisonCap (corrected)')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1.01),
               ncol=4,frameon=False,fontsize=7,columnspacing=1.4)
    for ax, (base, prot), title in zip(axes, PAIRS, ['(a) Capstone', '(b) CheriBSD']):
        for arm in (base, prot):
            runs = [r for r in rows if r['arm'] == arm]
            fractions = []
            for r in runs:
                cumulative = 0
                curve = []
                for count in r['bins']:
                    cumulative += count
                    curve.append(cumulative/r['allocations'])
                fractions.append(curve)
            y = [statistics.median(values) for values in zip(*fractions)]
            low = [min(values) for values in zip(*fractions)]
            high = [max(values) for values in zip(*fractions)]
            x = [(1<<(b+1))-1 for b in range(32)]
            ax.fill_between(x, low, high, step='post', color=COLOR[prot], alpha=.12)
            ax.step(x, y, where='post', color=COLOR[prot],
                    ls='--' if arm == base else '-', lw=2.2 if arm == base else 1.2,
                    label='Original' if arm == base else ('Sublet' if prot=='capstone-sublet' else 'PoisonCap (corrected)'))
        ax.set(xscale='log',xlim=(1,2**20),ylim=(0,1),title=title,
               xlabel='Allocation events since release',
               xticks=[1,16,256,4096,65536,1048576],
               xticklabels=['1','16','256','4K','64K','1M'])
        ax.grid(axis='both', color='.92', linewidth=.5)
    axes[0].set_ylabel('Same-start reuses / all allocations')
    for ext in ('pdf','svg','png'):
        fig.savefig(result/('reuse-gaps.'+ext),dpi=220)
    plt.close(fig)
    (result/'provenance.json').write_text(json.dumps({
        'definition': 'same arena start; logical free to next successful allocation; floor(log2 gap) bins; successful measured-unit memsys5 allocations denominator; setup prefix disclosed',
        'observer_invariance': 'Every SQLite phase ledger field except separately charged observer bytes equals the preceding normalized campaign in all 12 churn runs',
        'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs_sha256': {str(p.relative_to(scratch)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [scratch/'inputs.json',scratch/'run-cheri.py',scratch/'nodes4m/run-capstone.py']},
        'measured_binaries_sha256': {arm: hashlib.sha256((scratch/(f'{arm}-build/sqlite_silicon.dom' if arm.startswith('capstone') else arm)).read_bytes()).hexdigest() for arm in ARMS}
    }, indent=2)+'\n')
    print(json.dumps({a: {'allocations': [r['allocations'] for r in rows if r['arm']==a],
                          'reuses': [r['reuses'] for r in rows if r['arm']==a]}
                      for a in ARMS}, indent=2))


if __name__ == '__main__':
    main()
