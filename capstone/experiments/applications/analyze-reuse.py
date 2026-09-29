#!/usr/bin/env python3
"""Summarize application allocation behavior; keep failed attempts visible.

Inputs are runs.jsonl files from the existing application runners. This reports
address starts and allocator ledgers, not physical fragmentation or total RSS.
Control files test whether every recorded memory phase survived a platform change.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

from allocation_metrics import valid_allocations


def workload(point):
    return tuple(point[k] for k in ('application', 'size', 'batches', 'retained'))


def read(paths, arm):
    records = []
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['point']['arm'] != arm:
                raise ValueError(f'wrong platform in {path}')
            records.append(row)
    keys = [(workload(r['point']), r['repetition']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate workload/repetition within a platform')
    return records


def check(row):
    if row['status'] != 'pass':
        return
    phases = row['point']['expected_phases']
    # Reuse the runner's accounting checks when reading stored results.
    lines = ['EXP-ALLOC ' + ' '.join(f'{k}={v}' for k, v in s.items())
             for s in row['allocations']]
    if not valid_allocations('\n'.join(lines), phases):
        raise ValueError('invalid recorded allocation accounting')
    if [s['phase'] for s in row['memory']] != phases:
        raise ValueError('incomplete allocator phase sequence')
    if row['allocations'][-1]['failures']:
        raise ValueError('allocation failures need separate analysis')


def extract(row):
    point = row['point']
    result = dict(workload=list(workload(point)), arm=point['arm'],
                  repetition=row['repetition'], status=row['status'])
    if row['status'] != 'pass':
        return result
    end = row['allocations'][-1]
    metrics = {k: end[k] for k in ('allocations', 'unique', 'reused', 'peak', 'peak_blocks')}
    metrics['unique_per_peak_block'] = end['unique'] / end['peak_blocks']
    cumulative = 0
    for name in ('reuse1', 'reuse8', 'reuse64', 'reuse512', 'reuse4096', 'reuse_more'):
        cumulative += end[name]
        metrics[name + '_fraction'] = cumulative / end['allocations']
    result['metrics'] = metrics
    field = 'live' if point['arm'] == 'capstone-sublet' else 'allocated'
    result['ledger'] = 'occupied buddy blocks' if field == 'live' else 'jemalloc allocated'
    result['phases'] = [dict(phase=a['phase'], requested=a['live'], ledger=m[field],
                             allocations=a['allocations'], unique=a['unique'],
                             live_blocks=a['blocks'], reused=a['reused'])
                        for a, m in zip(row['allocations'], row['memory'])]
    released = [p for p in result['phases'] if p['phase'].startswith('released-')]
    if released:
        metrics['released_ledger_first'] = released[0]['ledger']
        metrics['released_ledger_last'] = released[-1]['ledger']
        metrics['released_ledger_min'] = min(p['ledger'] for p in released)
        metrics['released_ledger_max'] = max(p['ledger'] for p in released)
        metrics['released_requested_last'] = released[-1]['requested']
        metrics['released_ledger_growth'] = released[-1]['ledger'] - released[0]['ledger']
    return result


def invariance(controls, primary):
    groups = defaultdict(list)
    for row in primary:
        if row['status'] == 'pass':
            groups[workload(row['point'])].append(row)
    checks = []
    for old in controls:
        if old['status'] != 'pass':
            continue
        check(old)
        candidates = groups[workload(old['point'])]
        if not candidates:
            raise ValueError('control has no passing primary counterpart')
        # Compare all primary repeats, not a hand-selected matching repeat.
        for new in candidates:
            fields = ('allocations', 'memory', 'stdout_sha256', 'image_sha256')
            differing = [key for key in fields if old[key] != new[key]]
            differing += ['point.'+key for key in ('argv', 'environment', 'expected_stdout', 'expected_phases')
                          if old['point'].get(key) != new['point'].get(key)]
            checks.append(dict(workload=list(workload(old['point'])),
                               control_repetition=old['repetition'],
                               primary_repetition=new['repetition'],
                               equal=not differing, differing=differing))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capstone', type=Path, nargs='+', required=True)
    parser.add_argument('--cheribsd', type=Path, nargs='+', required=True)
    parser.add_argument('--control', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cap = read(args.capstone, 'capstone-sublet')
    cheri = read(args.cheribsd, 'cheribsd-default')
    controls = [r for path in args.control for r in read([path], 'capstone-sublet')]
    for row in cap + cheri:
        check(row)
    attempts = [extract(r) for r in cap + cheri]
    groups = defaultdict(list)
    for row in attempts:
        groups[(tuple(row['workload']), row['arm'])].append(row)
    summary = []
    for (key, arm), rows in sorted(groups.items()):
        good = [r for r in rows if r['status'] == 'pass']
        metrics = {}
        if good:
            for metric in good[0]['metrics']:
                values = [r['metrics'][metric] for r in good]
                metrics[metric] = dict(median=median(values), min=min(values), max=max(values))
        summary.append(dict(workload=list(key), arm=arm, counts=dict(Counter(r['status'] for r in rows)), metrics=metrics))
    checked = invariance(controls, cap)
    args.out.mkdir(parents=True, exist_ok=True)
    def save(name, obj):
        (args.out/name).write_text(json.dumps(obj, indent=2)+'\n')
    save('summary.json', summary)
    with (args.out/'attempts.jsonl').open('w') as out:
        for row in attempts:
            out.write(json.dumps(row, separators=(',', ':'))+'\n')
    save('invariance.json', dict(control_attempts=len(controls),
                               control_statuses=dict(Counter(r['status'] for r in controls)),
                               controls=len([r for r in controls if r['status']=='pass']),
                               pair_checks=len(checked), equal=sum(r['equal'] for r in checked),
                               checks=checked))
    paths = args.capstone + args.cheribsd + args.control
    save('inputs.json', [{'file': '/'.join(p.parts[-2:]), 'bytes': p.stat().st_size,
                         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths])
    print(json.dumps(dict(attempts=len(attempts), counts=dict(Counter(r['status'] for r in attempts)),
                          invariant_pairs=sum(r['equal'] for r in checked), control_pairs=len(checked))))
    if any(not row['equal'] for row in checked):
        raise SystemExit('platform controls differ; inspect invariance.json before making invariance claims')


if __name__ == '__main__':
    main()
