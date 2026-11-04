#!/usr/bin/env python3
"""Negative controls for the unpinned fault-ahead: broken MODULES, not edited transcripts.

The CV_TEST_HOOKS build has parameters that each remove one guard of the unpinned window
(module/capstone_vm.c, test_skip_*). Three are run here: the clip to the demand fault's VMA, the stop
check between batches, and the stop at the first resident PTE. For each, this runs the step-hooks gate
on the configurations that exercise that guard, once with the unbroken hooks module and once with the
broken variant, and requires the named check to PASS on the unbroken module and FAIL on the broken
one. A variant whose check does not flip means the oracle cannot see that guard.

The fourth, test_skip_arena_clip, is not run: no scenario a gate can build isolates it. A domain
mapping is a whole VMA (or the VMA clip bounds the window at the same page), and the image's NONLIN
arena is the whole 4 MiB image data, so removing the arena clip changes no window. Run through the
gate, adaptclip's planned limit stayed inside its arena with the clip gone, because the VMA clip
bounded it at the same page. The guard is defence in depth that is untested.

  run-mutants.py --qemu Q --images I --adapter A --module HOOKS.ko --application pin.dom --work W
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STEP = HERE.parent / 'run-step-hooks.py'
# variant: (module parameter, configurations that exercise the guard, checks that must flip)
VARIANTS = {
    'vma_clip':      ('test_skip_vma_clip=1', 'adaptclip', ['adaptclip_clip_window_in_vma', 'adaptclip_clipx_window_in_vma']),
    'stop_check':    ('test_skip_stop_check=1', 'adaptres,resched1', ['adaptres_no_fault_while_pending', 'resched1_no_fault_while_pending']),
    'resident_stop': ('test_skip_resident_stop=1', 'adaptgaps', ['adaptgaps_window']),
}


def run(a, label, configs, args):
    work = a.work / label
    cmd = [sys.executable, str(STEP), '--unpinned', '--qemu', str(a.qemu), '--images', str(a.images),
           '--adapter', str(a.adapter), '--module', str(a.module), '--application', str(a.application), '--pthread-application', str(a.pthread_application),
           '--work', str(work), '--only', configs, '--module-args', args]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = work / 'result.json'
    return json.loads(result.read_text())['checks'] if result.is_file() else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('qemu', 'images', 'adapter', 'module', 'application', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--pthread-application', type=Path, required=True)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    failed = []
    baselines = {}
    for variant, (param, configs, flips) in VARIANTS.items():
        if configs not in baselines:
            baselines[configs] = run(a, 'unbroken-' + configs.replace(',', '+'), configs, '')
        base, broken = baselines[configs], run(a, variant, configs, param)
        for check in flips:
            ok_base = bool(base and base.get(check) is True)
            fires = bool(broken and broken.get(check) is False)
            print(f"{'PASS' if ok_base and fires else 'FAIL'} {variant}: {check} unbroken={base and base.get(check)} "
                  f"broken={broken and broken.get(check)}")
            if not (ok_base and fires):
                failed.append(f'{variant}:{check}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
