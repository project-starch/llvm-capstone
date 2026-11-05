#!/usr/bin/env python3
"""Negative control for the unpinned no-change bound (fault_repeat_max): a broken module, not an edited transcript.

The unpinned fault path refuses a fault it has resolved 64 times at one (address, cause) with nothing
changed. A program that maps a page, touches it and unmaps it again and again gets the same address back, so
every round's first fault repeats that key in a NEW mapping, and the bound must not count it. mallocng does
exactly this when its only group is freed and the next allocation opens one (the runtime gate's sparse-churn
and sparse-stale cases died of it in ON mode once mallocng replaced the registry heap).

The CV_TEST_HOOKS build has `test_skip_repeat_arena`, which counts repeats by (address, cause) alone, as the
module did before. This boots the contract image's `map-churn` mode (300 rounds of map, touch, unmap) twice
with the same hooks module and requires

  ok       no module arguments          the mode exits 0, prints VIRTUAL_MAP_CHURN_OK with the address reused in more
                                        rounds than the bound, and the module logs no unpinned-loop refusal
  broken   test_skip_repeat_arena=1     the domain dies (exit 139) and the module logs the unpinned-loop refusal

on complete runs (the guest finished, the module loaded in the unpinned memory mode on both sides).

  run-repeat-control.py --qemu Q --images I --adapter A --module HOOKS.ko --application contract.dom --work W
  run-repeat-control.py --judge W        (read the two serial logs of an earlier run)
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import memory_mode  # noqa: E402

VARIANTS = (('ok', ''), ('broken', 'test_skip_repeat_arena=1'))
BOUND = 64
SCRIPT = '''#!/bin/sh
cd /mnt/vm || exit 1
insmod capstone_vm.ko %s || exit 1
%s
export CAPSTONE_VM_CONTRACT=environment
./capstone-vexec contract.dom map-churn 2>&1
echo MC_EXIT:$?
dmesg | grep 'capstone_vm:' | sed 's/^/KMSG:/'
rmmod capstone_vm
echo MC_CLEANUP:$?
echo VIRTUAL_STAGED_DONE
'''


def judge(work):
    logs = {}
    for name, _ in VARIANTS:
        try:
            logs[name] = (work / name / 'serial.log').read_text(errors='replace')
        except OSError:
            return False, [f'{name}: no serial log']
    problems = []
    for name, _ in VARIANTS:
        log = logs[name]
        for need in ('VIRTUAL_STAGED_DONE', 'MC_CLEANUP:0'):
            if need not in log:
                problems.append(f'{name}: the guest did not finish ({need} missing)')
        if not memory_mode.verified(log, True, 1):
            problems.append(f'{name}: the unpinned memory mode was not verified')
        if 'thread create refused' in log:
            problems.append(f'{name}: thread create refused')
    refused = {n: len(re.findall(r'resolve refused at unpinned-loop', l)) for n, l in logs.items()}
    ok_done = re.search(r'^VIRTUAL_MAP_CHURN_OK rounds=(\d+) same_address=(\d+)\s*$', logs['ok'], re.M)
    if 'MC_EXIT:0' not in logs['ok'] or not ok_done:
        problems.append('ok: the map-churn mode did not exit 0 with VIRTUAL_MAP_CHURN_OK')
    elif int(ok_done.group(2)) <= BOUND:
        problems.append(f'ok: the address was reused in {ok_done.group(2)} rounds, not more than the bound {BOUND}: the control is vacuous')
    if refused['ok']:
        problems.append(f'ok: {refused["ok"]} unpinned-loop refusals on the unbroken module')
    if 'MC_EXIT:139' not in logs['broken'] or 'VIRTUAL_MAP_CHURN_OK' in logs['broken']:
        problems.append('broken: the domain did not die (exit 139) on the broken module')
    if not refused['broken']:
        problems.append('broken: the module logged no unpinned-loop refusal')
    return not problems, problems + [f'unpinned-loop refusals: ok {refused["ok"]}, broken {refused["broken"]}']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--judge', type=Path)
    for name in ('qemu', 'images', 'adapter', 'module', 'application', 'work'):
        p.add_argument('--' + name, type=Path)
    a = p.parse_args()
    if a.judge:
        work = a.judge
    else:
        for name in ('qemu', 'images', 'adapter', 'module', 'application', 'work'):
            if getattr(a, name) is None:
                p.error('--' + name + ' is required')
        work = a.work
        work.mkdir(parents=True, exist_ok=False)
        for name, args in VARIANTS:
            stage = work / (name + '-stage')
            stage.mkdir()
            shutil.copyfile(a.module, stage / 'capstone_vm.ko')
            shutil.copyfile(a.adapter / 'capstone-vexec', stage / 'capstone-vexec')
            (stage / 'capstone-vexec').chmod(0o755)
            shutil.copyfile(a.application, stage / 'contract.dom')
            module_args = (memory_mode.args(True) + (' ' + args if args else '')).strip()
            (stage / 'gate.sh').write_text(SCRIPT % (module_args, memory_mode.echo('')))
            cmd = [sys.executable, str(HERE.parent / 'run-staged.py'), '--qemu', str(a.qemu), '--images', str(a.images),
                   '--stage', str(stage), '--work', str(work / name), '--disk-mib', '64', '--timeout', '600']
            env = dict(os.environ)
            env['CAPSTONE_QEMU_CPU_EXTRA'] = 'x-capstone-test-csrs=false,x-capstone-supervisor-slots=32,x-capstone-raw-moves=true'
            env.pop('GATE_UNPINNED_FAST_FAULT', None)
            subprocess.run(cmd, stdout=(work / (name + '.log')).open('w'), stderr=subprocess.STDOUT, env=env)
    ok, lines = judge(work)
    for line in lines:
        print(('PASS ' if ok else 'FAIL ') + line)
    print('repeat-control ' + ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
