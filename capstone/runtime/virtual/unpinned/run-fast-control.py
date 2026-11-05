#!/usr/bin/env python3
"""Negative control for the direct fault path (unpinned_fast_fault): a broken module, not an edited transcript.

The CV_TEST_HOOKS build has `test_fast_noflags`, which makes the direct path resolve a fault as a read
(FAULT_FLAG_WRITE dropped), so a store to a page that is not yet present maps the zero page, faults again, and
the domain dies. This runs the step-hooks gate on `base1` and `adapt` three ways with the same hooks module:

  fast-on-ok       unpinned_fast_fault=1                       every check of the two configurations passes
  fast-on-broken   unpinned_fast_fault=1 test_fast_noflags=1   the exit checks of all four runs fail
  fast-off-broken  test_fast_noflags=1 (fast path off)         every check equals fast-on-ok: the variant lives only in the path

and requires all of it, on complete runs (the result records' own bookkeeping checks true on every side). The
checks of the configurations that were not selected are false by construction under --only and are not read.

  run-fast-control.py --qemu Q --images I --adapter A --module HOOKS.ko --application pin.dom \\
      --pthread-application pthread.dom --work W
  run-fast-control.py --judge W        (read the three result records of an earlier run)
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STEP = HERE.parent / 'run-step-hooks.py'
VARIANTS = (('fast-on-ok', True, ''), ('fast-on-broken', True, 'test_fast_noflags=1'),
            ('fast-off-broken', False, 'test_fast_noflags=1'))
SELECTED = re.compile(r'^(base1|adapt)_')
EXITS = ('base1_walk_exit', 'base1_syscalls_exit', 'adapt_walk_exit', 'adapt_stream_exit')
RUN_CHECKS = ('completed', 'all_runs', 'all_reports', 'memory_mode_verified', 'no_contract_failure', 'cleanup')


def judge(work):
    records = {}
    for name, fast, args in VARIANTS:
        try:
            records[name] = json.loads((work / name / 'result.json').read_text())
        except (OSError, ValueError):
            return False, [f'{name}: no result record']
    problems = []
    for name, fast, args in VARIANTS:
        r = records[name]
        checks = r.get('checks') if isinstance(r.get('checks'), dict) else {}
        if r.get('module_args', '') != args or r.get('unpinned') is not True:
            problems.append(f'{name}: module_args {r.get("module_args")!r} unpinned {r.get("unpinned")!r}')
        problems += [f'{name}: {k} is {checks.get(k)}' for k in RUN_CHECKS if checks.get(k) is not True]
    mine = {n: {k: v for k, v in records[n].get('checks', {}).items() if SELECTED.match(k)} for n, _, _ in VARIANTS}
    ok, broken, off = mine['fast-on-ok'], mine['fast-on-broken'], mine['fast-off-broken']
    if not ok or any(v is not True for v in ok.values()):
        problems.append(f'fast-on-ok: checks not true: {sorted(k for k, v in ok.items() if v is not True)}')
    if set(ok) != set(broken) or set(ok) != set(off):
        problems.append('the three records do not hold the same checks')
    flipped = sorted(k for k in broken if broken.get(k) is not True and ok.get(k) is True)
    if any(broken.get(k) is not False for k in EXITS) or any(ok.get(k) is not True for k in EXITS):
        problems.append(f'exit checks {EXITS} did not flip: {[(k, ok.get(k), broken.get(k)) for k in EXITS]}')
    if off != ok:
        problems.append(f'fast-off-broken differs from fast-on-ok: {sorted(k for k in ok if off.get(k) != ok[k])}')
    return not problems, problems + [f'{len(flipped)} checks flip on the broken direct path: {flipped}']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--judge', type=Path)
    for name in ('qemu', 'images', 'adapter', 'module', 'application', 'pthread-application', 'work'):
        p.add_argument('--' + name, type=Path)
    a = p.parse_args()
    if a.judge:
        work = a.judge
    else:
        for name in ('qemu', 'images', 'adapter', 'module', 'application', 'pthread_application', 'work'):
            if getattr(a, name) is None:
                p.error('--' + name.replace('_', '-') + ' is required')
        work = a.work
        work.mkdir(parents=True, exist_ok=False)
        for name, fast, args in VARIANTS:
            env = dict(os.environ)
            env.pop('GATE_UNPINNED_FAST_FAULT', None)
            if fast:
                env['GATE_UNPINNED_FAST_FAULT'] = '1'
            cmd = [sys.executable, str(STEP), '--unpinned', '--qemu', str(a.qemu), '--images', str(a.images),
                   '--adapter', str(a.adapter), '--module', str(a.module), '--application', str(a.application),
                   '--pthread-application', str(a.pthread_application), '--work', str(work / name),
                   '--only', 'base1,adapt', '--module-args', args]
            # The gate exits 1 whenever a check is false, which --only makes certain; the record decides.
            subprocess.run(cmd, stdout=(work / (name + '.log')).open('w'), stderr=subprocess.STDOUT, env=env)
    ok, lines = judge(work)
    for line in lines:
        print(('PASS ' if ok else 'FAIL ') + line)
    print('fast-control ' + ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
