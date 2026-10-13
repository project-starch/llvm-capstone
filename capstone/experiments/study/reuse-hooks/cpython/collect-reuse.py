#!/usr/bin/env python3
"""reuse.json from the rep.json files of run-cpython-reuse.py / run-mruby-reuse.py.

    collect-reuse.py --program NAME --workload TEXT --out reuse.json \
        --rep stock=PATH/rep.json [--rep stock=...] --rep protected=PATH/rep.json [...] \
        [--images stock=PATH protected=PATH] [--extra KEY=JSON_FILE ...]

Bucket k = observer bins[k-1] (distance d with 2^(k-1) <= d < 2^k); empty buckets omitted. With
more than one repetition per arm, "buckets" and the counters are repetition 1's and every
repetition is listed under "repetitions". Refuses a repetition whose run did not complete, whose
oracle failed (exit status), or whose observer error is nonzero. "comparison" holds, per boundary,
the median bucket per arm and the share of reuses whose bucket differs between the arms,
sum_k |s_k/N_s - p_k/N_p| / 2, for repetition 1 against repetition 1 and over all pairs.
"""
import argparse
import hashlib
import json
from pathlib import Path

COUNTERS = ('issues', 'releases', 'reuses', 'distinct_starts', 'attempts', 'error')


def buckets(bins):
    return {str(i + 1): n for i, n in enumerate(bins) if n}


def median_bucket(bins):
    total = sum(bins)
    if not total:
        return None
    run = 0
    for i, n in enumerate(bins):
        run += n
        if 2 * run >= total:
            return i + 1


def reusediff(a, b):
    na, nb = sum(a), sum(b)
    return sum(abs(x / na - y / nb) for x, y in zip(a, b)) / 2


def arm_record(reps, boundary):
    out = []
    for i, rep in enumerate(reps, 1):
        rg = rep['reuse_gap'].get(boundary)
        if not rep['completed'] or rep['rc'] != 0 or rg is None or rg['error'] != 0:
            raise SystemExit(f'{rep["arm"]} rep {rep["rep"]}: rejected (completed={rep["completed"]} '
                             f'rc={rep["rc"]} reuse_gap={rg and rg["error"]})')
        if rg['issues'] != rg['attempts']:
            raise SystemExit(f'{rep["arm"]} rep {rep["rep"]}: issues != attempts')
        r = {'buckets': buckets(rg['bins']), **{k: rg[k] for k in COUNTERS},
             'capacity': rg['capacity'], 'median_bucket': median_bucket(rg['bins']),
             'guest_seconds': rep.get('guest_seconds'), 'boot_wall_seconds': rep.get('boot_wall_seconds')}
        out.append((r, rg['bins']))
    first = dict(out[0][0])
    if len(out) > 1:
        first['repetitions'] = [r for r, _ in out]
        first['buckets_from'] = 'repetition 1'
    return first, [b for _, b in out]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--program', required=True)
    p.add_argument('--workload', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rep', action='append', required=True)
    p.add_argument('--image', action='append', default=[], help='ARM=PATH of the image to hash')
    p.add_argument('--extra', action='append', default=[], help='KEY=JSON_FILE merged in at the top level')
    a = p.parse_args()
    reps = {'stock': [], 'protected': []}
    for item in a.rep:
        arm, path = item.split('=', 1)
        reps[arm].append(json.loads(Path(path).read_text()))
    for arm in reps:
        reps[arm].sort(key=lambda r: r['rep'])
        if not reps[arm]:
            raise SystemExit(f'no repetition for {arm}')
    names = sorted(set(reps['stock'][0]['reuse_gap']) | set(reps['protected'][0]['reuse_gap']))
    if not names:
        raise SystemExit('no REUSE-GAP boundary in the repetitions')
    result = {'program': a.program, 'pre_registration': '16d2171a3edc', 'workload': a.workload,
              'boundaries': {}, 'comparison': {}}
    for name in names:
        rec, bins = {}, {}
        for arm in ('stock', 'protected'):
            rec[arm], bins[arm] = arm_record(reps[arm], name)
        result['boundaries'][name] = rec
        pairs = [reusediff(s, q) for s in bins['stock'] for q in bins['protected']]
        within = {arm: [reusediff(x, y) for i, x in enumerate(bins[arm]) for y in bins[arm][i + 1:]]
                  for arm in bins}
        result['comparison'][name] = {
            'median_bucket': {arm: [median_bucket(b) for b in bins[arm]] for arm in bins},
            'reusediff_rep1': reusediff(bins['stock'][0], bins['protected'][0]),
            'reusediff_all_pairs': [min(pairs), max(pairs)],
            'reusediff_within_arm': {arm: ([min(v), max(v)] if v else None) for arm, v in within.items()},
            'identical_bins_rep1': bins['stock'][0] == bins['protected'][0],
        }
    result['images'] = {}
    for item in a.image:
        arm, path = item.split('=', 1)
        result['images'][arm] = {'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    for arm in ('stock', 'protected'):
        result['images'].setdefault(arm, {})['staged_sha256'] = reps[arm][0]['staged_sha256']
    r0 = reps['stock'][0]
    result['platform'] = {'platform_sha256': r0['platform_sha256'], 'run_staged_sha256': r0['run_staged_sha256']}
    for item in a.extra:
        key, path = item.split('=', 1)
        result[key] = json.loads(Path(path).read_text())
    a.out.write_text(json.dumps(result, indent=1) + '\n')
    for name, c in result['comparison'].items():
        print(name, json.dumps(c))


if __name__ == '__main__':
    main()
