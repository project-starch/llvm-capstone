#!/usr/bin/env python3
"""Negative controls for the unpinned fault-ahead: broken MODULES, not edited transcripts.

The CV_TEST_HOOKS build has parameters that each remove one guard of the unpinned window
(module/capstone_vm.c, test_skip_*). Three are run here: the clip to the demand fault's VMA, the stop
check between batches, and the stop at the first resident PTE. For each, this runs the step-hooks gate
on the configurations that exercise that guard, once with the unbroken hooks module and once with the
broken variant, and requires the named check to PASS on the unbroken module and FAIL on the broken
one. A variant whose check does not flip means the oracle cannot see that guard.

A flip only counts when BOTH runs are real runs. A check is false as readily because the guest never
got that far (the module failed to load, the guest halted early, the transcript is empty) as because
the guard is gone, so for each side this also requires: the result record exists and carries the
arguments of that side; the guest run completed; every selected configuration reported and ran in every
mode it has, with its own exit check true; the memory mode was verified; no contract failure and a
clean cleanup; and the observations the named check reads are present (the windows and events of the
run it judges, the fault counters of the report it judges). The other checks of the gate are not
required: --only leaves the unselected configurations absent on purpose, so their checks are false.

The fourth, test_skip_arena_clip, is not run: no scenario a gate can build isolates it. A domain
mapping is a whole VMA (or the VMA clip bounds the window at the same page), and the image's NONLIN
arena is the whole 4 MiB image data, so removing the arena clip changes no window. Run through the
gate, adaptclip's planned limit stayed inside its arena with the clip gone, because the VMA clip
bounded it at the same page. The guard is defence in depth that is untested.

  run-mutants.py --qemu Q --images I --adapter A --module HOOKS.ko --application pin.dom \\
      --pthread-application pthread.dom --work W
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STEP = HERE.parent / 'run-step-hooks.py'
# variant: (module parameter, configurations that exercise the guard, checks that must flip, the
# observations behind them: {run 'config:mode': fields that must be non-empty lists},
# {report 'config': counters that must be present})
VARIANTS = {
    'vma_clip': ('test_skip_vma_clip=1', 'adaptclip',
                 ['adaptclip_clip_window_in_vma', 'adaptclip_clipx_window_in_vma'],
                 {'adaptclip:clip': ('raw_windows', 'raw_events'), 'adaptclip:clipx': ('raw_windows', 'raw_events')},
                 {'adaptclip': ('fault_requests', 'entries')}),
    'stop_check': ('test_skip_stop_check=1', 'adaptres,resched1',
                   ['adaptres_no_fault_while_pending', 'resched1_no_fault_while_pending'],
                   {'adaptres:walk': ('raw_windows', 'raw_events'), 'resched1:walk': ('raw_events',)},
                   {'adaptres': ('fault_pending', 'fault_requests', 'entries'),
                    'resched1': ('fault_pending', 'fault_requests', 'entries')}),
    'resident_stop': ('test_skip_resident_stop=1', 'adaptgaps', ['adaptgaps_window'],
                      {'adaptgaps:gaps': ('raw_windows', 'ahead')},
                      {'adaptgaps': ('fault_requests', 'entries')}),
}
# the gate's own bookkeeping checks: each must be true on a run that counts
RUN_CHECKS = ('completed', 'all_runs', 'all_reports', 'memory_mode_verified', 'no_contract_failure', 'cleanup')


def step_hooks():
    sys.path.insert(0, str(STEP.parent))
    spec = importlib.util.spec_from_file_location('run_step_hooks', STEP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def side_problems(result, configs, args, runs_seen, reports_seen, configurations=None):
    """Why a step-hooks result record is not a usable run of `configs` with module arguments `args`."""
    if not isinstance(result, dict):
        return ['no result record']
    configurations = configurations or step_hooks().CONFIGS
    problems = []
    checks = result.get('checks') if isinstance(result.get('checks'), dict) else {}
    runs = result.get('runs') if isinstance(result.get('runs'), dict) else {}
    reports = result.get('reports') if isinstance(result.get('reports'), dict) else {}
    if result.get('module_args', '') != args:
        problems.append(f"module_args {result.get('module_args')!r}, want {args!r}")
    if result.get('unpinned') is not True:
        problems.append('not an unpinned run')
    problems += [f'{name} is {checks.get(name)}' for name in RUN_CHECKS if checks.get(name) is not True]
    names = configs.split(',')
    if sorted(reports) != sorted(names):
        problems.append(f'reports {sorted(reports)}, want {sorted(names)}')
    want_runs = sorted(f'{n}:{m}' for n in names for m in configurations[n][1].split())
    if sorted(runs) != want_runs:
        problems.append(f'runs {sorted(runs)}, want {want_runs}')
    for key in want_runs:
        n, m = key.split(':')
        if checks.get(f'{n}_{m}_exit') is not True:
            problems.append(f'{n}_{m}_exit is {checks.get(f"{n}_{m}_exit")}')
    for key, fields in runs_seen.items():
        run = runs.get(key)
        if not isinstance(run, dict) or run.get('region') is None:
            problems.append(f'run {key} has no region')
            continue
        problems += [f'run {key} has no {f}' for f in fields if not run.get(f)]
    for name, fields in reports_seen.items():
        report = reports.get(name)
        if not isinstance(report, dict):
            problems.append(f'report {name} missing')
            continue
        problems += [f'report {name} has no {f}' for f in fields if not isinstance(report.get(f), int)]
        if isinstance(report.get('fault_requests'), int) and report['fault_requests'] <= 0:
            problems.append(f'report {name}: fault_requests is 0')
    return problems


def judge(variant, base, broken, configurations=None):
    """[(passed, text)] for the checks `variant` must flip, given the unbroken and the broken result."""
    param, configs, flips, runs_seen, reports_seen = VARIANTS[variant]
    base_problems = side_problems(base, configs, '', runs_seen, reports_seen, configurations)
    broken_problems = side_problems(broken, configs, param, runs_seen, reports_seen, configurations)
    verdicts = []
    for check in flips:
        base_value = (base or {}).get('checks', {}).get(check) if isinstance(base, dict) else None
        broken_value = (broken or {}).get('checks', {}).get(check) if isinstance(broken, dict) else None
        ok = base_value is True and broken_value is False and not base_problems and not broken_problems
        text = f'{variant}: {check} unbroken={base_value} broken={broken_value}'
        if base_problems:
            text += f' [unbroken run unusable: {"; ".join(base_problems[:4])}]'
        if broken_problems:
            text += f' [broken run unusable: {"; ".join(broken_problems[:4])}]'
        verdicts.append((ok, text))
    return verdicts


def run(a, label, configs, args):
    work = a.work / label
    cmd = [sys.executable, str(STEP), '--unpinned', '--qemu', str(a.qemu), '--images', str(a.images),
           '--adapter', str(a.adapter), '--module', str(a.module), '--application', str(a.application), '--pthread-application', str(a.pthread_application),
           '--work', str(work), '--only', configs, '--module-args', args]
    # The gate exits 1 whenever any check is false, which --only makes certain; the record decides.
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = work / 'result.json'
    try:
        return json.loads(result.read_text()) if result.is_file() else None
    except ValueError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('qemu', 'images', 'adapter', 'module', 'application', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--pthread-application', type=Path, required=True)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    failed = []
    baselines = {}
    for variant, (param, configs, flips, _, _) in VARIANTS.items():
        if configs not in baselines:
            baselines[configs] = run(a, 'unbroken-' + configs.replace(',', '+'), configs, '')
        broken = run(a, variant, configs, param)
        for ok, text in judge(variant, baselines[configs], broken):
            print(('PASS ' if ok else 'FAIL ') + text)
            if not ok:
                failed.append(text.split(' unbroken=')[0])
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
