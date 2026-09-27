#!/usr/bin/env python3
"""Plan paired application studies and feed the existing persistent-guest runners.

No VM lifecycle, benchmark download, or automatic retry lives here. A plan fixes
the denominator before any runs; local bindings identify qualified artifacts.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import re

HERE = Path(__file__).resolve().parent
ARMS = ('capstone', 'capstone-sublet', 'cheribsd-revocation-on', 'cheribsd-revocation-off')
COMPARISONS = {
    'cheribsd': ARMS,
    'nested-poisoncap': ('capstone', 'capstone-sublet', 'poisoncap-spatial', 'poisoncap-temporal'),
}
ARM_PLATFORMS = {arm: arm.split('-')[0] for group in COMPARISONS.values() for arm in group}
POISONCAP_GATE = ('PoisonCap application execution is not yet qualified: require matched '
                 'allocator boundaries, observed inner policy, quarantine-path accounting, '
                 'and pinned kernel/libc/SDK identities. Component replays do not qualify.')


def arms(plan):
    return COMPARISONS[plan.get('comparison', 'cheribsd')]


def platform(arm):
    if arm not in ARM_PLATFORMS:
        raise ValueError('unknown study arm: '+arm)
    return ARM_PLATFORMS[arm]


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def make_plan(catalog, profile, suites, repeats, seed, matrix=None, comparison='cheribsd'):
    if profile not in ('nested', 'outer-malloc') or repeats < 1:
        raise ValueError('explicit profile and positive repetitions required')
    if comparison not in COMPARISONS or (comparison == 'nested-poisoncap' and profile != 'nested'):
        raise ValueError('PoisonCap comparison requires the nested profile')
    selected_arms = COMPARISONS[comparison]
    guests = tuple(dict.fromkeys(platform(a) for a in selected_arms))
    known = {s['id']: s for s in catalog['suites']}
    if not suites or len(set(suites)) != len(suites) or set(suites) - known.keys():
        raise ValueError('select unique known suites')
    workloads = {}
    selected_cases = {name+'/'+case for name in suites for case in known[name]['cases']}
    if matrix is not None and set(matrix) != selected_cases:
        raise ValueError('matrix must explicitly cover every selected suite case')
    for name in suites:
        suite = known[name]
        for case in suite['cases']:
            key = name + '/' + case
            variants = matrix[key] if matrix is not None else {'upstream': {'work': 'upstream-default'}}
            if not variants or any(not parameters for parameters in variants.values()):
                raise ValueError('each case needs named variants with explicit work parameters')
            for variant, parameters in variants.items():
                if not re.fullmatch('[a-zA-Z0-9_-]+', variant):
                    raise ValueError('invalid variant name')
                workloads[key+'@'+variant] = dict(suite=suite, case=case, variant=variant,
                    parameters=parameters, application=catalog['applications'][suite['application']])
    rng = random.Random(seed)
    cells = []
    # Repeat is the block. Both arms of a platform share one boot; record the
    # shuffled order, and alternate platform order between repetition blocks.
    for repeat in range(repeats):
        for guest in (guests if repeat % 2 == 0 else tuple(reversed(guests))):
            block = [dict(workload=key, arm=arm, platform=guest, repetition=repeat)
                     for key in workloads for arm in selected_arms if platform(arm) == guest]
            rng.shuffle(block)
            for cell in block:
                cell['id'] = identity(cell)[:20]
                cells.append(cell)
    result = dict(schema_version=1, profile=profile, seed=seed, repeats=repeats,
                  catalog_sha256=identity(catalog), workloads=workloads, cells=cells)
    if comparison != 'cheribsd':
        result.update(schema_version=2, comparison=comparison)
    result['plan_id'] = identity(result)
    return result


def check_plan(plan):
    body = {k: v for k, v in plan.items() if k != 'plan_id'}
    if identity(body) != plan['plan_id']:
        raise ValueError('plan changed after its identity was assigned')


def qualified_points(plan, key, binding):
    """Check four artifacts as one comparison, never qualify just the winner."""
    if plan.get('comparison', 'cheribsd') == 'nested-poisoncap':
        raise ValueError(POISONCAP_GATE)
    spec = plan['workloads'][key]
    suite = spec['suite']
    if binding['source'] != suite['source']:
        raise ValueError('benchmark source does not match the planned pin')
    if not any(suite['source'].get(k) for k in ('revision', 'sha256', 'sha3_256')):
        raise ValueError('benchmark source must be pinned before qualification')
    if binding['application_version'] != spec['application']['version']:
        raise ValueError('application version mismatch')
    if binding['profile'] != plan['profile'] or set(binding['arms']) != set(ARMS):
        raise ValueError('all four arms of the same profile are required')
    if binding['parameters'] != spec['parameters'] or not binding['adaptation']:
        raise ValueError('work parameters must match the plan; adaptation description required')
    if not re.fullmatch('[0-9a-f]{64}', binding['oracle_reference_sha256']):
        raise ValueError('independent reference output hash is required')
    oracle = binding['oracle']
    if not isinstance(oracle['expected_stdout'], str) or not oracle['expected_phases']:
        raise ValueError('explicit output and phase oracles are required')
    if (file_hash(binding['oracle_reference']) != binding['oracle_reference_sha256'] or
            hashlib.sha256(oracle['expected_stdout'].encode()).hexdigest() != binding['oracle_reference_sha256']):
        raise ValueError('oracle disagrees with the preserved reference output')
    points, manifests = {}, {}
    common_inputs = binding['input_sha256']
    for arm in ARMS:
        item = binding['arms'][arm]
        build = read(item['build_manifest'])
        manifests[arm] = build
        digest = file_hash(item['binary'])
        if digest != build['image_sha256'] or not build.get('allocations'):
            raise ValueError('binary differs from its instrumented build manifest')
        if build.get('application', build.get('app')) != suite['application']:
            raise ValueError('build manifest names another application')
        if {role: file_hash(path) for role, path in item['inputs'].items()} != common_inputs:
            raise ValueError('workload inputs differ between platforms or changed on disk')
        if not item['resources'] or not item['qualification_evidence']:
            raise ValueError('resource accounting and adapter qualification evidence required')
        point = copy.deepcopy(item['point'])
        if any(k.startswith(('_RUNTIME_', 'MALLOC_', 'EXP_CHERI_')) for k in point['environment']):
            raise ValueError('binding cannot override the named arm policy')
        if point['application'] != suite['application']:
            raise ValueError('point names another application')
        point.update(arm=arm, allocations=True, **oracle)
        if platform(arm) == 'capstone':
            expected_heap = 'sublet' if arm == 'capstone-sublet' and plan['profile'] == 'outer-malloc' else 'level0'
            expected_nested = suite['application'] if arm == 'capstone-sublet' and plan['profile'] == 'nested' else 'none'
            if (build['heap'], build['nested']) != (expected_heap, expected_nested):
                raise ValueError('outer heap / internal allocator scope does not match the arm')
            point['image'] = item['binary']
        else:
            point['revocation'] = int(arm.endswith('-on'))
            guest_binary = point['files'].get(item['binary'])
            if not guest_binary or point['argv'][0] != guest_binary:
                raise ValueError('CheriBSD argv does not execute the identified binary')
        point['study_artifact'] = dict(binary_sha256=digest,
            build_manifest_sha256=file_hash(item['build_manifest']), resources=item['resources'],
            qualification_evidence=item['qualification_evidence'])
        points[arm] = point
    on, off = (points[a] for a in ARMS[2:])
    for field in ('argv', 'environment', 'files', 'study_artifact'):
        if on[field] != off[field]:
            raise ValueError('CheriBSD on/off must use identical artifacts and workload settings')
    if len({m['allocations_sha256'] for m in manifests.values()}) != 1:
        raise ValueError('allocation observer differs between arms')
    return points


def attempts(plan, paths):
    known = {c['id']: c for c in plan['cells']}
    found = {}
    bindings = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            meta = row['point']['study']
            key = meta['cell_id']
            if meta['plan_id'] != plan['plan_id'] or key not in known:
                raise ValueError('attempt belongs to another plan')
            if key in found:
                raise ValueError('duplicate attempt; retries require a separately identified plan')
            if row['point']['arm'] != known[key]['arm']:
                raise ValueError('attempt arm disagrees with its planned cell')
            cell = known[key]
            if any(meta[k] != cell[k] for k in ('workload', 'repetition')) or meta['profile'] != plan['profile']:
                raise ValueError('attempt workload or repetition disagrees with its planned cell')
            previous = bindings.setdefault(cell['workload'], meta['binding_sha256'])
            if previous != meta['binding_sha256']:
                raise ValueError('workload artifacts changed during the campaign')
            found[key] = row
    return found


def emit(plan, bindings, guest, rows, repeat=None):
    points, blocked = [], {}
    qualified = {}
    check_bindings(rows, bindings)
    if repeat is not None and not 0 <= repeat < plan['repeats']:
        raise ValueError('repetition is outside the plan')
    for key in plan['workloads']:
        if key not in bindings:
            blocked[key] = (POISONCAP_GATE if plan.get('comparison') == 'nested-poisoncap'
                            else 'No qualified four-arm binding')
            continue
        # Invalid supplied evidence is an error, never an unavailable benchmark.
        qualified[key] = qualified_points(plan, key, bindings[key])
    for cell in plan['cells']:
        if cell['platform'] != guest or (repeat is not None and cell['repetition'] != repeat):
            continue
        if cell['id'] in rows or cell['workload'] in blocked:
            continue
        point = copy.deepcopy(qualified[cell['workload']][cell['arm']])
        point['id'] = 'study-' + cell['id']
        point['study'] = dict(plan_id=plan['plan_id'], cell_id=cell['id'],
            workload=cell['workload'], repetition=cell['repetition'], profile=plan['profile'],
            binding_sha256=identity(bindings[cell['workload']]))
        points.append(point)
    return points, blocked


def check_bindings(rows, bindings):
    for row in rows.values():
        meta = row['point']['study']
        if meta['workload'] not in bindings or identity(bindings[meta['workload']]) != meta['binding_sha256']:
            raise ValueError('cannot resume or report against changed bindings')


def summary(plan, rows, bindings):
    result = []
    for key in plan['workloads']:
        for arm in arms(plan):
            counts = Counter()
            for cell in plan['cells']:
                if cell['workload'] != key or cell['arm'] != arm:
                    continue
                row = rows.get(cell['id'])
                counts[row['status'] if row else ('pending' if key in bindings else 'unqualified')] += 1
            result.append(dict(workload=key, arm=arm, counts=dict(counts)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    listing = sub.add_parser('list')
    listing.add_argument('--catalog', type=Path, default=HERE/'catalog.json')
    planning = sub.add_parser('plan')
    planning.add_argument('--catalog', type=Path, default=HERE/'catalog.json')
    planning.add_argument('--profile', choices=['nested', 'outer-malloc'], required=True)
    planning.add_argument('--comparison', choices=COMPARISONS, default='cheribsd',
                          help='PoisonCap supports planning only until application qualification')
    planning.add_argument('--suites', nargs='+', required=True)
    planning.add_argument('--repeat', type=int, default=3)
    planning.add_argument('--seed', type=int, default=1)
    planning.add_argument('--matrix', type=Path, help='Explicit variant and work parameters for every selected case')
    planning.add_argument('--out', type=Path, required=True)
    for name in ('points', 'status'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--plan', type=Path, required=True)
        cmd.add_argument('--bindings', type=Path, required=True)
        cmd.add_argument('--runs', type=Path, nargs='*', default=[])
        if name == 'points':
            cmd.add_argument('--platform', choices=['capstone', 'cheribsd', 'poisoncap'], required=True)
            cmd.add_argument('--repetition', type=int)
            cmd.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'list':
        for suite in read(args.catalog)['suites']:
            print(f"{suite['id']}: {len(suite['cases'])} cases; {suite['adaptation_status']}; {suite['kind']}")
        return
    if args.command == 'plan':
        plan = make_plan(read(args.catalog), args.profile, args.suites, args.repeat, args.seed,
                         read(args.matrix) if args.matrix else None, args.comparison)
        write_new(args.out, plan)
        print(f"{len(plan['cells'])} planned attempts; plan {plan['plan_id']}")
        return
    plan, bindings = read(args.plan), read(args.bindings)
    check_plan(plan)
    rows = attempts(plan, args.runs)
    if args.command == 'points':
        points, blocked = emit(plan, bindings, args.platform, rows, args.repetition)
        write_new(args.out, points)
        print(json.dumps(dict(ready=len(points), unqualified=blocked), indent=2))
    else:
        check_bindings(rows, bindings)
        for key, binding in bindings.items():
            qualified_points(plan, key, binding)
        print(json.dumps(summary(plan, rows, bindings), indent=2))


if __name__ == '__main__':
    main()
