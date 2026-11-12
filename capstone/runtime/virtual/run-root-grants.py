#!/usr/bin/env python3
"""Unit test of root grants: root-contract.c in one boot, with the module loaded root_grants=1.

The processor needs supervisor references: set CAPSTONE_QEMU_CPU_EXTRA to the CPU properties of
the memory mode plus x-capstone-supervisor-refs=true.

--control runs the same program against a module whose retire() skips the CREVOKE. The stale
access must then succeed and read the new mapping, so the run passes only if no fault occurs.
--root-grants 0 loads the module without the parameter: the grant is linear, and the stale access
still faults with 25, through CSRETIRE.
--no-refs runs on a processor without supervisor references. The module must not load. Where the
firmware hands cause 24 to S the load is refused with a message. The gate platform's monitor keeps
the trap and halts the machine at the module's probe, after the module printed why.

The second program run is mode `stale`: the program ends a mapping itself and unmaps it, once
directly and once after the adapter collected identities. The kernel's copy of the reference is
stale in both, and the unmap must succeed without a kernel fault.
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
CHECKS = ('grant_reads_and_writes', 'libc_accepts_copyable_grant', 'other_mapping_survives_unmap')
STALE_CHECKS = ('unmap_after_own_release', 'unmap_after_own_release_and_collection')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('adapter', 'application', 'qemu', 'images', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--module', type=Path, help='module to load instead of the adapter\'s (the control)')
    p.add_argument('--root-grants', type=int, choices=(0, 1), default=1)
    p.add_argument('--control', action='store_true', help='the module skips the CREVOKE: the stale access must succeed')
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
    (stage / 'gate.sh').write_text(f'''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko{module_args}
echo ROOT_INSMOD:$?
{memory_mode.echo('ROOT_')}
echo "ROOT_GRANTS_PARAMETER:$(cat /sys/module/capstone_vm/parameters/root_grants 2>/dev/null || echo A)"
./capstone-vexec root.dom
echo ROOT_EXIT:$?
CAPSTONE_VM_STATS=1 ./capstone-vexec root.dom stale
echo ROOT_STALE_EXIT:$?
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

    def line(pattern):
        m = re.search(pattern, log, re.M)
        return m.group(1) if m else None

    insmod, status = line(r'^ROOT_INSMOD:(\d+)$'), line(r'^ROOT_EXIT:(\d+)$')
    causes = [int(n) for n in re.findall(r'domain fault cause=(\d+)', log)]
    tests = {}
    outcome = None
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
        tests['grant_type'] = line(r'^ROOT_GRANT_TYPE:(\d+)$') == str(a.root_grants)
        for name in CHECKS:
            tests[name] = f'ROOT_CHECK:{name}:PASS' in log
        tests['remapped_at_the_same_address'] = 'ROOT_REMAP same_address=1' in log
        tests['no_contract_failure'] = 'ROOT_FAIL' not in log
        if a.control:
            tests['control_stale_access_reads_new_mapping'] = 'ROOT_STALE_READ value=0x77' in log
            tests['control_no_fault'] = causes == [] and status == '0'
        else:
            tests['stale_access_faults_25'] = causes == [25] and status == '139'
            tests['stale_access_did_not_read'] = 'ROOT_STALE_READ' not in log
        for name in STALE_CHECKS:
            tests[name] = f'ROOT_CHECK:{name}:PASS' in log
        tests['stale_mode_exit_zero'] = line(r'^ROOT_STALE_EXIT:(\d+)$') == '0'
        collections = line(r'^CAPSTONE_VM_STATS .*\bcollections=(\d+)')
        tests['stale_mode_collected'] = collections is not None and int(collections) >= 1
        tests['module_unloaded'] = 'ROOT_CLEANUP:0' in log
        tests['no_kernel_fault'] = 'Oops' not in log and 'Kernel panic' not in log

    result = dict(status='PASS' if all(tests.values()) else 'FAIL', root_grants=a.root_grants,
                  control=a.control, no_refs=a.no_refs, unpinned=a.unpinned,
                  passed=sum(tests.values()), total=len(tests), tests=tests,
                  insmod=insmod, exit=status, fault_causes=causes, no_refs_outcome=outcome)
    (a.work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'passed', 'total', 'exit', 'fault_causes')}))
    for name, ok in tests.items():
        if not ok:
            print('FAIL', name)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
