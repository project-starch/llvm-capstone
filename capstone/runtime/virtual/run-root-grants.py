#!/usr/bin/env python3
"""Unit test of root grants: root-contract.c in one boot, one run per mode, with the module loaded
root_grants=1 (or 0 with --root-grants 0).

The processor needs supervisor references: set CAPSTONE_QEMU_CPU_EXTRA to the CPU properties of
the memory mode plus x-capstone-supervisor-refs=true.

Modes (root-contract.c lists the checks of each):
  unmap   munmap through the mapping's reference, a second munmap, the old reference at the
          reused address; ends with a stale read that must fault with 25 (exit 139)
  refuse  an integer address, an object over the whole page, a partial munmap, a reference the
          program ended itself, the same after a collection; nothing faults (exit 0)
  narrow  a reference narrowed to part of the mapping (exit 0)
  renew   the reference from before CRENEW, then the renewed one; ends with a fault (139)
  mremap  the libc's map-copy-unmap mremap; ends with a fault through the old reference (139)
Every refusal of the module is counted (parameter unmap_refused) and reported in the kernel log
with its reason, which this runner reads back.

--root-grants 0 loads the module without the parameter. The checks that hold only with root
grants print ROOT_SKIP with a reason, and the runner requires those skips instead.
--control runs against a module whose unmap check skips condition 4 (the bounds cover the
mapping). The narrowed reference must then be accepted: the narrow mode fails at that check
(exit 1) and every other mode passes as without the control.
--no-refs runs on a processor without supervisor references. The module must not load. Where the
firmware hands cause 24 to S the load is refused with a message. The gate platform's monitor keeps
the trap and halts the machine at the module's probe, after the module printed why.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import memory_mode

HERE = Path(__file__).resolve().parent
MODES = ('unmap', 'refuse', 'narrow', 'renew', 'mremap')
# Per mode: the checks, those that need root grants (skipped otherwise), and the expected exit.
CHECKS = {
    'unmap': ('grant_reads_and_writes', 'libc_accepts_copyable_grant',
              'unmap_through_mapping_reference', 'other_mapping_survives_unmap',
              'second_unmap_refused', 'stale_reference_at_reused_address_refused'),
    'refuse': ('address_refused', 'nested_object_refused', 'partial_unmap_refused',
               'stale_reference_refused', 'stale_reference_after_collection_refused'),
    'narrow': ('narrowed_reference_refused', 'whole_reference_after_narrowed'),
    'renew': ('renewed_old_reference_refused', 'renewed_mapping_unmaps'),
    'mremap': ('public_mremap_moves', 'mremap_old_reference_refused'),
}
ROOT_ONLY = {'stale_reference_at_reused_address_refused', 'address_refused',
             'nested_object_refused', 'stale_reference_refused',
             'stale_reference_after_collection_refused', 'narrowed_reference_refused',
             'whole_reference_after_narrowed', 'renewed_old_reference_refused'}
FAULTING = {'unmap', 'renew', 'mremap'}
# The kernel's reason for each refusal that reaches the module (capstone_vm.c cv_unmap_reasons).
# The others (a second munmap, a partial munmap, the old reference after mremap) are refused by
# the launcher, which no longer finds a whole mapping at that address.
REASONS = {
    'not a reference': 'address_refused',
    'not live': 'stale_reference_refused',
    'not the mapping': 'nested_object_refused',
    'does not cover the mapping': 'narrowed_reference_refused',
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('adapter', 'application', 'qemu', 'images', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--module', type=Path, help='module to load instead of the adapter\'s (the control)')
    p.add_argument('--root-grants', type=int, choices=(0, 1), default=1)
    p.add_argument('--control', action='store_true',
                   help='the module skips the covering-bounds check: the narrowed munmap must be accepted')
    p.add_argument('--no-refs', action='store_true', help='the processor has no supervisor references: insmod must fail')
    p.add_argument('--unpinned', action='store_true', help='load the module with unpinned=1')
    p.add_argument('--timeout', type=int, default=300)
    a = p.parse_args()

    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    shutil.copy2(a.adapter / 'capstone-vexec', stage / 'capstone-vexec')
    shutil.copy2(a.module or a.adapter / 'module/capstone_vm.ko', stage / 'capstone_vm.ko')
    shutil.copy2(a.application, stage / 'root.dom')
    module_args = memory_mode.args(a.unpinned).replace(' root_grants=1', '')
    if a.root_grants:
        module_args += ' root_grants=1'
    runs = '\n'.join(f'''echo ROOT_MODE_BEGIN:{mode}
CAPSTONE_VM_STATS=1 ./capstone-vexec root.dom {mode}
echo ROOT_EXIT:{mode}:$?''' for mode in MODES)
    (stage / 'gate.sh').write_text(f'''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko{module_args}
echo ROOT_INSMOD:$?
{memory_mode.echo('ROOT_')}
echo "ROOT_GRANTS_PARAMETER:$(cat /sys/module/capstone_vm/parameters/root_grants 2>/dev/null || echo A)"
{runs}
echo "ROOT_UNMAP_REFUSED:$(cat /sys/module/capstone_vm/parameters/unmap_refused 2>/dev/null || echo A)"
dmesg | grep 'capstone_vm: unmap refused' | sed 's/^.*capstone_vm: /ROOT_KERNEL:/'
rmmod capstone_vm
echo ROOT_CLEANUP:$?
echo VIRTUAL_STAGED_DONE
''')
    subprocess.run([sys.executable, str(HERE / 'run-staged.py'), '--qemu', str(a.qemu.resolve()),
                    '--images', str(a.images.resolve()), '--stage', str(stage.resolve()),
                    '--work', str((a.work / 'guest').resolve()), '--timeout', str(a.timeout),
                    '--disk-mib', '64'])
    serial = a.work / 'guest/serial.log'
    log = serial.read_text(errors='replace').replace('\r', '') if serial.exists() else ''

    def line(pattern, text=log):
        m = re.search(pattern, text, re.M)
        return m.group(1) if m else None

    insmod = line(r'^ROOT_INSMOD:(\d+)$')
    causes = [int(n) for n in re.findall(r'domain fault cause=(\d+)', log)]
    tests = {}
    outcome = None
    exits = {}
    if a.no_refs:
        refused = insmod not in (None, '0') and 'root_grants needs supervisor references' in log
        halted = insmod is None and 'Print = Scalar(0x18)' in log and 'VIRTUAL_STAGED_DONE' not in log
        outcome = 'refused' if refused else 'halted at the probe' if halted else 'unexpected'
        tests['probe_announced'] = 'Probing supervisor references' in log
        tests['module_not_loaded'] = 'capstone_vm: loaded' not in log
        tests['refused_or_halted_at_the_probe'] = refused or halted
        tests['no_kernel_oops'] = 'Oops' not in log and 'Kernel panic' not in log
    else:
        tests['module_loaded'] = insmod == '0'
        tests['parameter_reads_back'] = line(r'^ROOT_GRANTS_PARAMETER:(\w+)$') == ('Y' if a.root_grants else 'N')
        tests['memory_mode_verified'] = memory_mode.verified(log, a.unpinned, 1)
        for mode in MODES:
            begin = log.find(f'ROOT_MODE_BEGIN:{mode}\n')
            end = log.find(f'ROOT_EXIT:{mode}:', begin)
            part = log[begin:end] if begin >= 0 and end >= 0 else ''
            status = line(rf'^ROOT_EXIT:{mode}:(\d+)$')
            exits[mode] = status
            mode_causes = [int(n) for n in re.findall(r'domain fault cause=(\d+)', part)]
            tests[f'{mode}:grant_type'] = line(r'^ROOT_GRANT_TYPE:(\d+)$', part) == str(a.root_grants)
            control_cut = a.control and mode == 'narrow'
            for name in CHECKS[mode]:
                if a.root_grants or name not in ROOT_ONLY:
                    tests[f'{mode}:{name}'] = f'ROOT_CHECK:{name}:PASS' in part
                else:
                    tests[f'{mode}:{name}:skipped'] = re.search(rf'^ROOT_SKIP:{name}:\S', part, re.M) is not None
            if control_cut:
                # The control accepts the narrowed reference: its check must fail, nothing else.
                tests['narrow:narrowed_reference_refused'] = not tests['narrow:narrowed_reference_refused']
                tests['narrow:whole_reference_after_narrowed'] = not tests['narrow:whole_reference_after_narrowed']
                tests['narrow:control_accepts_narrowed_reference'] = \
                    'ROOT_FAIL:munmap through a narrowed reference' in part and status == '1'
                tests['narrow:control_no_fault'] = mode_causes == []
                continue
            tests[f'{mode}:no_contract_failure'] = 'ROOT_FAIL' not in part
            if mode in FAULTING:
                tests[f'{mode}:remapped_at_the_same_address'] = 'ROOT_REMAP same_address=1' in part
                tests[f'{mode}:stale_access_faults_25'] = mode_causes == [25] and status == '139'
                tests[f'{mode}:stale_access_did_not_read'] = 'ROOT_STALE_READ' not in part
            else:
                tests[f'{mode}:exit_zero'] = mode_causes == [] and status == '0'
            if mode == 'refuse' and a.root_grants:
                collections = line(r'^CAPSTONE_VM_STATS .*\bcollections=(\d+)', part)
                tests['refuse:collected'] = collections is not None and int(collections) >= 1
        reported = re.findall(r'^ROOT_KERNEL:unmap refused \(([^)]*)\)', log, re.M)
        refused_count = line(r'^ROOT_UNMAP_REFUSED:(\w+)$')
        if a.root_grants:
            for reason, check in REASONS.items():
                if a.control and reason == 'does not cover the mapping':
                    tests['kernel_reports:control_never_refuses_cover'] = reason not in reported
                else:
                    tests[f'kernel_reports:{reason}'] = reason in reported
            tests['kernel_count_matches_reports'] = refused_count == str(len(reported))
        else:
            tests['kernel_refuses_nothing'] = refused_count == '0' and reported == []
        tests['module_unloaded'] = 'ROOT_CLEANUP:0' in log
        tests['no_kernel_fault'] = 'Oops' not in log and 'Kernel panic' not in log

    result = dict(status='PASS' if all(tests.values()) else 'FAIL', root_grants=a.root_grants,
                  control=a.control, no_refs=a.no_refs, unpinned=a.unpinned,
                  passed=sum(tests.values()), total=len(tests), tests=tests,
                  insmod=insmod, exits=exits, fault_causes=causes, no_refs_outcome=outcome)
    (a.work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'passed', 'total', 'exits', 'fault_causes')}))
    for name, ok in tests.items():
        if not ok:
            print('FAIL', name)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
