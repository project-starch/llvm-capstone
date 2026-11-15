#!/usr/bin/env python3
"""The PHP reuse-distance record (docs/plans/2026-11-15-php-reuse-distance.md) from its run bundles.

    collect.py RUNS_DIR OUT.json

RUNS_DIR holds one trap_platform.py run bundle per arm and repetition, named ARM-REP
(stock-1 .. exact-3), each with results.json and one step named php-ARM. Every step must exit 0 and
print, per boundary, one REUSE-GAP line with error=0 and one REUSE-GAP-PHP line. The record gives,
per arm and boundary, the counters, buckets (bucket k = bins[k-1]) and median bucket of each
repetition, and per boundary the reusediff share (sum over k of |a_k/N_a - b_k/N_b| / 2) of each
arm against stock: repetition 1 against 1, the range over all pairs, and the range within each arm.
It also prints the checks of the plan's predictions 1, 3 and 4.
"""
import json
import re
import sys
from pathlib import Path

ARMS = ('stock', 'protected', 'exact')
BOUNDARIES = ('zend-small', 'zend-large')
COUNTERS = ('attempts', 'issues', 'releases', 'reuses', 'distinct_starts', 'drops', 'fresh_reuses', 'error')
EXTRA = ('fresh', 'cache_hits', 'moves', 'resizes', 'shutdown_releases')
TIMING = re.compile(r'^(\S.*?)( +)(\d[\d,]*\.\d{3})$')


def main():
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    reps = {arm: sorted(runs.glob(f'{arm}-*'), key=lambda p: int(p.name.rsplit('-', 1)[1])) for arm in ARMS}
    record = {'plan': 'capstone/docs/plans/2026-11-15-php-reuse-distance.md', 'arms': {}, 'pairs': {}}
    bins = {}
    for arm, dirs in reps.items():
        if not dirs:
            continue
        readings = [read_run(d, arm) for d in dirs]
        record['arms'][arm] = readings
        bins[arm] = {b: [r['boundaries'][b]['bins'] for r in readings] for b in BOUNDARIES}
    for arm in ARMS[1:]:
        if arm not in bins or 'stock' not in bins:
            continue
        record['pairs'][arm] = {b: pair_record(bins['stock'][b], bins[arm][b]) for b in BOUNDARIES}
    record['within_arm'] = {arm: {b: within(bins[arm][b]) for b in BOUNDARIES} for arm in bins}
    out.write_text(json.dumps(record, indent=1) + '\n')
    report(record)


def read_run(directory, arm):
    results = json.loads((directory / 'results.json').read_text())
    step = next(r for r in results if r['name'] == f'php-{arm}')
    if step['kind'] != 'exit' or step['value'] != 0:
        raise SystemExit(f'{directory.name}: rejected, the step ended {step["kind"]} {step["value"]}')
    text = step['output']
    boundaries = {}
    for b in BOUNDARIES:
        line = re.search(rf'^REUSE-GAP {b} (.*)$', text, re.M)
        extra = re.search(rf'^REUSE-GAP-PHP {b} (.*)$', text, re.M)
        if not line or not extra:
            raise SystemExit(f'{directory.name}: no REUSE-GAP or REUSE-GAP-PHP line for {b}')
        fields = dict(word.split('=', 1) for word in line.group(1).split())
        reading = {k: int(fields[k]) for k in COUNTERS}
        reading.update({k: int(v) for k, v in (w.split('=', 1) for w in extra.group(1).split())})
        reading['bins'] = [int(x) for x in fields['bins'].split(',')]
        reading['buckets'] = {k + 1: n for k, n in enumerate(reading['bins']) if n}
        reading['median_bucket'] = median_bucket(reading['bins'])
        if reading['error']:
            raise SystemExit(f'{directory.name}: rejected, {b} error={reading["error"]}')
        boundaries[b] = reading
    program = [line for line in text.splitlines() if not line.startswith(('REUSE-GAP', 'ZEND_CAP_MODE', 'PHP_VIRTUAL'))]
    widths = [len(m.group(3)) for m in map(TIMING.match, program) if m]
    shape = [TIMING.sub(r'\1 <t>', line) for line in program]
    return {'run': directory.name, 'boundaries': boundaries, 'timing_widths': widths, 'output_shape': shape}


def median_bucket(bins):
    total = sum(bins)
    run = 0
    for i, n in enumerate(bins):
        run += n
        if total and 2 * run >= total:
            return i + 1
    return None


def reusediff(a, b):
    na, nb = sum(a), sum(b)
    return sum(abs(x / na - y / nb) for x, y in zip(a, b)) / 2 if na and nb else None


def pair_record(stock, other):
    pairs = [reusediff(s, o) for s in stock for o in other]
    return {'reusediff_rep1': reusediff(stock[0], other[0]), 'reusediff_all_pairs': [min(pairs), max(pairs)]}


def within(reps):
    values = [reusediff(x, y) for i, x in enumerate(reps) for y in reps[i + 1:]]
    return [min(values), max(values)] if values else None


def report(record):
    arms = record['arms']
    for b in BOUNDARIES:
        print(f'== {b}')
        for arm, readings in arms.items():
            for r in readings:
                x = r['boundaries'][b]
                print(f"  {r['run']:12} issues={x['issues']} releases={x['releases']} reuses={x['reuses']} "
                      f"distinct={x['distinct_starts']} error={x['error']} median={x['median_bucket']} "
                      f"cache_hits={x['cache_hits']} fresh={x['fresh']} moves={x['moves']} resizes={x['resizes']} "
                      f"shutdown={x['shutdown_releases']} drops={x['drops']}")
        for arm, pairs in record['pairs'].items():
            print(f"  reusediff {arm} vs stock: rep1 {pairs[b]['reusediff_rep1']}, all pairs {pairs[b]['reusediff_all_pairs']}")
        for arm, w in record['within_arm'].items():
            print(f"  reusediff within {arm}: {w[b]}")
    print('== checks')
    shapes = {tuple(r['output_shape']) for readings in arms.values() for r in readings}
    print(f'  prediction 4, output equal apart from timings: {len(shapes) == 1}')
    if 'stock' in arms and 'protected' in arms:
        for b in BOUNDARIES:
            for s, p in zip(arms['stock'], arms['protected']):
                x, y = s['boundaries'][b], p['boundaries'][b]
                same_width = s['timing_widths'] == p['timing_widths']
                ok = (x['cache_hits'] == y['cache_hits'] and x['fresh'] - x['moves'] == y['fresh'] - y['moves']
                      and x['shutdown_releases'] == y['shutdown_releases'] and y['resizes'] == 0
                      and y['moves'] == x['moves'] + x['resizes'])
                print(f"  prediction 1, {b} {s['run']}/{p['run']} (same timing widths: {same_width}): {ok}")
    if 'protected' in arms and 'exact' in arms:
        for b in BOUNDARIES:
            for p, e in zip(arms['protected'], arms['exact']):
                x, y = p['boundaries'][b], e['boundaries'][b]
                same = all(x[k] == y[k] for k in COUNTERS + EXTRA) and x['bins'] == y['bins']
                same_width = p['timing_widths'] == e['timing_widths']
                print(f"  prediction 3, {b} {p['run']}/{e['run']} (same timing widths: {same_width}): {same}")


if __name__ == '__main__':
    main()
