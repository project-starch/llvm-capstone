#!/usr/bin/env python3
"""Host test of run-mutants.py's judgement: a broken module only counts as a negative control when both
runs are real runs that reached the check's observations. No guest.

Synthetic result records of the step-hooks gate (the fields run-mutants reads), one good pair that must
pass for every variant and then one planted defect at a time, each of which must fail the variant. The
last case is the reviewer's: the real evaluator's output on an EMPTY transcript (every named target is
false, as it would be after a module that never loaded) in place of the broken-run record.

  python3 capstone/runtime/virtual/unpinned/test-mutants.py [SAVED_RUN_DIR]

With a directory of a saved `unpinned-mutants` gate (run-arm.sh GATES=unpinned-mutants), the real
records of that run are judged as well and must pass.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('run_mutants', HERE / 'run-mutants.py')
mutants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mutants)
STEP = mutants.step_hooks()
CONFIGS = STEP.CONFIGS


def record(variant, broken, **edit):
    """A complete step-hooks result for the configurations of `variant`, the named checks true
    (unbroken) or false (broken)."""
    param, configs, flips, runs_seen, reports_seen = mutants.VARIANTS[variant]
    names = configs.split(',')
    checks = {k: True for k in mutants.RUN_CHECKS}
    runs = {}
    for n in names:
        for m in CONFIGS[n][1].split():
            checks[f'{n}_{m}_exit'] = True
            runs[f'{n}:{m}'] = dict(region=4096, raw_windows=[1], raw_events=[1], ahead=[1])
    reports = {n: dict(fault_pending=0, fault_requests=9, entries=9) for n in names}
    for flip in flips:
        checks[flip] = not broken
    result = dict(checks=checks, runs=runs, reports=reports, unpinned=True, module_args=param if broken else '')
    for key, value in edit.items():
        result[key] = value
    return result


def verdict(variant, base, broken):
    return all(ok for ok, _ in mutants.judge(variant, base, broken))


def main():
    failed = total = 0

    def case(name, variant, base, broken, want):
        nonlocal failed, total
        total += 1
        got = verdict(variant, base, broken)
        ok = got == want
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {variant}: {name}: judged {'pass' if got else 'fail'}, want {'pass' if want else 'fail'}")

    for variant in mutants.VARIANTS:
        good, bad = record(variant, False), record(variant, True)
        case('genuine pair', variant, good, bad, True)
        # the named check does not flip: the oracle cannot see the guard
        case('broken module, check still true', variant, good, record(variant, False, module_args=bad['module_args']), False)
        case('unbroken module already false', variant, bad, bad, False)
        case('no result record on the broken side', variant, good, None, False)
        case('no result record on the unbroken side', variant, None, bad, False)
        for key in mutants.RUN_CHECKS:
            for side in ('broken', 'unbroken'):
                b, u = copy.deepcopy(bad), copy.deepcopy(good)
                (b if side == 'broken' else u)['checks'][key] = False
                case(f'{side} run: {key} false', variant, u, b, False)
        b = copy.deepcopy(bad); del b['checks']['memory_mode_verified']
        case('broken run: memory_mode_verified missing', variant, good, b, False)
        case('broken run carries the unbroken arguments', variant, good, record(variant, True, module_args=''), False)
        case('unbroken run carries the broken arguments', variant, record(variant, False, module_args=bad['module_args']), bad, False)
        case('broken run is not unpinned', variant, good, record(variant, True, unpinned=False), False)
        b = copy.deepcopy(bad); b['reports'] = {}
        case('broken run: no report at all', variant, good, b, False)
        b = copy.deepcopy(bad); b['runs'] = {}
        case('broken run: no run at all', variant, good, b, False)
        b = copy.deepcopy(bad); b['runs'][next(iter(b['runs']))]['exit'] = 139
        b['checks'][f"{next(iter(b['runs'])).replace(':', '_')}_exit"] = False
        case('broken run: a scenario ended in the wrong status', variant, good, b, False)
        b = copy.deepcopy(bad); b['reports'][next(iter(b['reports']))]['fault_requests'] = 0
        case('broken run: the module resolved no fault', variant, good, b, False)
        b = copy.deepcopy(bad); del b['reports'][next(iter(b['reports']))]['entries']
        case('broken run: a counter of the report is missing', variant, good, b, False)
        key = next(iter(mutants.VARIANTS[variant][3])); field = mutants.VARIANTS[variant][3][key][-1]
        b = copy.deepcopy(bad); b['runs'][key][field] = []
        case('broken run: no events recorded', variant, good, b, False)
        b = copy.deepcopy(bad); next(iter(b['runs'].values()))['region'] = None
        case('broken run: no region', variant, good, b, False)
        u = copy.deepcopy(good); u['runs'][key][field] = []
        case('unbroken run: no events recorded', variant, u, bad, False)
        # --only: the checks of the configurations that were not selected do not exist, and need not
        b = copy.deepcopy(bad); b['checks']['base1_walk_exit'] = False; b['checks']['adapt_walk_window'] = False
        case('checks of unselected configurations false (--only)', variant, good, b, True)

        # the reviewer's case: the evaluator on an empty transcript, with the selected configurations
        configs = mutants.VARIANTS[variant][1].split(',')
        empty = STEP.evaluate('', configs, True, False, True)[0]
        assert empty['completed'] is False and empty['all_runs'] is False and empty['all_reports'] is False
        flips = mutants.VARIANTS[variant][2]
        assert all(empty[f] is False for f in flips), 'the empty transcript must make every named target false'
        case('empty transcript as the broken run (module never loaded)', variant, good,
             dict(checks=empty, runs={}, reports={}, unpinned=True, module_args=bad['module_args']), False)
        case('empty transcript on both sides', variant,
             dict(checks=empty, runs={}, reports={}, unpinned=True, module_args=''),
             dict(checks=empty, runs={}, reports={}, unpinned=True, module_args=bad['module_args']), False)

    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
        for variant, (param, configs, flips, _, _) in mutants.VARIANTS.items():
            def load(name):
                f = root / name / 'result.json'
                return json.loads(f.read_text()) if f.is_file() else None
            case(f'saved run {root.name}', variant, load('unbroken-' + configs.replace(',', '+')), load(variant), True)
    print(f'{total - failed}/{total} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
