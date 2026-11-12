#!/usr/bin/env python3
"""Unit test of root grants: root-contract.c in one boot, one run per mode, with the module loaded
root_grants=1 (or 0 with --root-grants 0).

The processor needs supervisor references: set CAPSTONE_QEMU_CPU_EXTRA to the CPU properties of
the memory mode plus x-capstone-supervisor-refs=true.

Modes (root-contract.c lists the checks of each):
  unmap        munmap through the mapping's reference, a second munmap, the old reference at the
               reused address; ends with a stale read that must fault with 25 (exit 139)
  refuse       an integer address, an object over the whole page, a partial munmap, a reference
               the program ended itself, the same after a collection; nothing faults (exit 0)
  narrow       a reference narrowed to part of the mapping (exit 0)
  renew        the reference from before CRENEW, then the renewed one; ends with a fault through
               an object derived in the renewed mapping (139)
  renew-ref    the same, and the fault is through the renewed reference itself (139)
  mremap       the libc's map-copy-unmap mremap; ends with a fault through the old reference (139)
  exit-victim  a thread exits naming a victim mapping it holds no reference to (exit 0)
  exit-renew   a thread renews its stack's mapping and exits; the main thread's copy of the renewed
               reference must fault with 25 (139)
  create       the frame's hand-over with the reference from before CRENEW (refused) and with the
               renewed one; the renewed reference must fault with 25 afterwards (139)
  create-fail  the launcher fails a create after the module took the frame: the frame's mapping is
               released (exit 0)
  detached     a detached pthread's stack is unmapped at its exit (exit 0)
Every refusal of a request is counted (parameter unmap_refused) and reported in the kernel log
with its reason and request, and so is every CV_RETIRE that the module refuses because the
mapping's reservation is live (parameter retire_refused). The runner reads both back per mode (the
log is read and cleared after each mode) and compares them with what each mode does.

--root-grants 0 loads the module without the parameter. The checks that hold only with root
grants print ROOT_SKIP with a reason, and the runner requires those skips instead.
--control NAME runs against a module that is one change away from the adapter's (--module):
  cover          the unmap check skips condition 4 (the bounds cover the mapping): the narrowed
                 munmap is accepted, and the narrow mode fails at that check (exit 1)
  parent         the check skips condition 3 (the operand's record is a child of M): the object
                 over the whole page ends the whole mapping, and the refuse mode fails there
  exit-operand   the stack's mapping at thread exit ends through the program's operand (the a1
                 slot, cleared with the frame by then) instead of M: the renewed stack lifetime
                 survives, exit-renew reads through it, and detached finds the stack still mapped
Every other mode must pass as without the control.
--launcher PATH runs another capstone-vexec. --launcher-bypass says that it lacks the launcher's
whole-mapping check: the partial munmap then reaches the module, which must refuse it
(reason "range is not the mapping").
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
MODES = ('unmap', 'refuse', 'narrow', 'renew', 'renew-ref', 'mremap',
         'exit-victim', 'exit-renew', 'create', 'create-fail', 'detached')
# Per mode: the checks, those that need root grants (skipped otherwise), and the expected exit.
CHECKS = {
    'unmap': ('grant_reads_and_writes', 'libc_accepts_copyable_grant',
              'unmap_through_mapping_reference', 'other_mapping_survives_unmap',
              'second_unmap_refused', 'stale_reference_at_reused_address_refused'),
    'refuse': ('address_refused', 'nested_object_refused', 'partial_unmap_refused',
               'stale_reference_refused', 'stale_reference_after_collection_refused'),
    'narrow': ('narrowed_reference_refused', 'whole_reference_after_narrowed'),
    'renew': ('renewed_old_reference_refused', 'renewed_mapping_unmaps'),
    'renew-ref': ('renewed_old_reference_refused', 'renewed_mapping_unmaps'),
    'mremap': ('public_mremap_moves', 'mremap_old_reference_refused'),
    'exit-victim': ('exit_with_own_stack_at_victim_refused', 'exit_with_victim_address_refused',
                    'victim_unmaps_afterwards'),
    'exit-renew': ('stack_renewed_before_exit',),
    'create': ('create_with_stale_frame_refused', 'create_with_renewed_frame'),
    'create-fail': ('failed_create_releases_frame',),
    'detached': ('detached_exit_unmaps_stack',),
}
ROOT_ONLY = {'stale_reference_at_reused_address_refused', 'address_refused',
             'nested_object_refused', 'stale_reference_refused',
             'stale_reference_after_collection_refused', 'narrowed_reference_refused',
             'whole_reference_after_narrowed', 'renewed_old_reference_refused',
             'exit_with_own_stack_at_victim_refused', 'exit_with_victim_address_refused',
             'victim_unmaps_afterwards', 'create_with_stale_frame_refused'}
FAULTING = {'unmap', 'renew', 'renew-ref', 'mremap', 'exit-renew', 'create'}
# Modes whose stale read does not map the address again first (the frame is not reused).
NO_REMAP = {'create'}
# The kernel's refusals per mode with root grants, as (reason, request) in order
# (capstone_vm.c cv_unmap_reasons, request_refused). The other refusals (a second munmap, the
# second page of a partial munmap, the old reference after mremap) are the launcher's, which no
# longer finds a whole mapping at that address. exit-victim's first exit names the victim with
# the own stack's reference moved there, the stack's end: a live reference to another
# reservation (its reason is filled in from the operand's type). A reason may be a tuple of
# alternatives: the collection in the refuse mode may clear the tag of the program's own stale
# reference before the munmap hands it over.
REFUSALS = {
    'unmap': [('not live', 'munmap')],
    'refuse': [('not a reference', 'munmap'), ('not the mapping', 'munmap'),
               ('not live', 'munmap'), ('not live', 'munmap'),
               (('not live', 'not a reference'), 'munmap')],
    'narrow': [('does not cover the mapping', 'munmap')],
    'renew': [('not live', 'munmap')],
    'renew-ref': [('not live', 'munmap')],
    'mremap': [],
    'exit-victim': [(None, 'thread exit'), ('not a reference', 'thread exit')],
    'exit-renew': [],
    'create': [('not live', 'thread create')],
    'create-fail': [],
    'detached': [],
}
# What each control changes: the mode it must break, and how.
CONTROL_CUTS = {
    'cover': {'narrow': ('ROOT_FAIL:munmap through a narrowed reference', 'narrowed_reference_refused')},
    'parent': {'refuse': ('ROOT_FAIL:munmap through an object over the whole page', 'nested_object_refused')},
    'exit-operand': {'exit-renew': ('ROOT_STALE_READ', None),
                     'detached': ('ROOT_FAIL:the stack\'s mapping is gone after the thread\'s exit',
                                  'detached_exit_unmaps_stack')},
}
# The kernel's refusals of the launcher's CV_RETIRE per mode: the two stacks that exit-victim's
# threads named stay mapped, and under the exit-operand control so do the stacks in its cut modes.
RETIRES = {'exit-victim': 2}
CONTROL_RETIRES = {'exit-operand': {'exit-renew': 1, 'detached': 1}}
REPORT = re.compile(r'^ROOT_KERNEL:([\w-]+):unmap refused \(([^)]*)\) at ([a-z ]+):', re.M)
RETIRE_REPORT = re.compile(r'^ROOT_KERNEL:([\w-]+):retire refused:', re.M)


def refusals_match(seen, expected):
    """seen matches expected entry by entry; an expected reason may be a tuple of alternatives."""
    if len(seen) != len(expected):
        return False
    for (reason, request), (want, want_request) in zip(seen, expected):
        wants = want if isinstance(want, tuple) else (want,)
        if reason not in wants or request != want_request:
            return False
    return True


def expected_refusals(mode, control, bypass, victim_reason):
    """The (reason, request) list a mode's kernel log must show, or None where a control cuts the
    mode short (then only a prefix is checked)."""
    refusals = list(REFUSALS[mode])
    if mode == 'exit-victim':
        refusals[0] = (victim_reason, 'thread exit')
    if bypass and mode == 'refuse':
        # The first page of the partial munmap reaches the module; the second page does not
        # start a mapping and the launcher still refuses it.
        refusals.insert(2, ('range is not the mapping', 'munmap'))
    if control == 'parent' and mode == 'refuse':
        return refusals[:1]                 # the address; the nested object is accepted
    if control == 'parent' and mode == 'exit-victim' and victim_reason == 'not the mapping':
        refusals[0] = ('does not cover the mapping', 'thread exit')
    if control == 'cover' and mode == 'narrow':
        return []
    return refusals


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('adapter', 'application', 'qemu', 'images', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--module', type=Path, help='module to load instead of the adapter\'s (a control)')
    p.add_argument('--launcher', type=Path, help='capstone-vexec to run instead of the adapter\'s')
    p.add_argument('--launcher-bypass', action='store_true',
                   help='the launcher lacks its whole-mapping check; the module must refuse a partial munmap')
    p.add_argument('--root-grants', type=int, choices=(0, 1), default=1)
    p.add_argument('--control', choices=sorted(CONTROL_CUTS), help='the module is this control (see above)')
    p.add_argument('--modes', default=','.join(MODES), help='comma-separated subset of the modes')
    p.add_argument('--no-refs', action='store_true', help='the processor has no supervisor references: insmod must fail')
    p.add_argument('--unpinned', action='store_true', help='load the module with unpinned=1')
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    modes = [m for m in a.modes.split(',') if m]
    if any(m not in MODES for m in modes):
        p.error(f'unknown mode in {a.modes}')
    if a.control and not a.module:
        p.error('--control needs --module')

    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    shutil.copy2(a.launcher or a.adapter / 'capstone-vexec', stage / 'capstone-vexec')
    shutil.copy2(a.module or a.adapter / 'module/capstone_vm.ko', stage / 'capstone_vm.ko')
    shutil.copy2(a.application, stage / 'root.dom')
    module_args = memory_mode.args(a.unpinned).replace(' root_grants=1', '')
    if a.root_grants:
        module_args += ' root_grants=1'
    runs = '\n'.join(f'''echo ROOT_MODE_BEGIN:{mode}
CAPSTONE_VM_STATS=1 ./capstone-vexec root.dom {mode}
echo ROOT_EXIT:{mode}:$?
dmesg -c | grep 'capstone_vm: \\(unmap\\|retire\\) refused' | sed 's/^.*capstone_vm: /ROOT_KERNEL:{mode}:/'
''' for mode in modes)
    (stage / 'gate.sh').write_text(f'''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko{module_args}
echo ROOT_INSMOD:$?
{memory_mode.echo('ROOT_')}
echo "ROOT_GRANTS_PARAMETER:$(cat /sys/module/capstone_vm/parameters/root_grants 2>/dev/null || echo A)"
dmesg -c > /dev/null
{runs}
echo "ROOT_UNMAP_REFUSED:$(cat /sys/module/capstone_vm/parameters/unmap_refused 2>/dev/null || echo A)"
echo "ROOT_RETIRE_REFUSED:$(cat /sys/module/capstone_vm/parameters/retire_refused 2>/dev/null || echo A)"
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
    refusals_seen = {}
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
        cuts = CONTROL_CUTS.get(a.control, {})
        reports = REPORT.findall(log)
        retire_reports = RETIRE_REPORT.findall(log)
        retires_seen = {}
        for mode in modes:
            begin = log.find(f'ROOT_MODE_BEGIN:{mode}\n')
            end = log.find(f'ROOT_EXIT:{mode}:', begin)
            part = log[begin:end] if begin >= 0 and end >= 0 else ''
            status = line(rf'^ROOT_EXIT:{mode}:(\d+)$')
            exits[mode] = status
            mode_causes = [int(n) for n in re.findall(r'domain fault cause=(\d+)', part)]
            seen = [(reason, request) for m, reason, request in reports if m == mode]
            refusals_seen[mode] = seen
            retires_seen[mode] = retire_reports.count(mode)
            if a.root_grants:
                retires = CONTROL_RETIRES.get(a.control, {}).get(mode, RETIRES.get(mode, 0))
                tests[f'{mode}:kernel_retire_refusals'] = retires_seen[mode] == retires
            tests[f'{mode}:grant_type'] = line(r'^ROOT_GRANT_TYPE:(\d+)$', part) == str(a.root_grants)
            if mode in cuts:
                # The control must break this mode, at this point, and nothing else.
                marker, check = cuts[mode]
                for name in CHECKS[mode]:
                    if name == check:
                        tests[f'{mode}:{name}:control_fails'] = f'ROOT_CHECK:{name}:PASS' not in part
                        break
                    tests[f'{mode}:{name}'] = f'ROOT_CHECK:{name}:PASS' in part
                tests[f'{mode}:control_breaks_here'] = marker in part
                tests[f'{mode}:control_no_fault'] = mode_causes == []
                if a.root_grants:
                    expected = expected_refusals(mode, a.control, a.launcher_bypass, None)
                    tests[f'{mode}:kernel_refusals'] = refusals_match(seen[:len(expected)], expected)
                continue
            for name in CHECKS[mode]:
                if a.root_grants or name not in ROOT_ONLY:
                    tests[f'{mode}:{name}'] = f'ROOT_CHECK:{name}:PASS' in part
                else:
                    tests[f'{mode}:{name}:skipped'] = re.search(rf'^ROOT_SKIP:{name}:\S', part, re.M) is not None
            tests[f'{mode}:no_contract_failure'] = 'ROOT_FAIL' not in part
            if mode in FAULTING:
                if mode not in NO_REMAP:
                    tests[f'{mode}:remapped_at_the_same_address'] = 'ROOT_REMAP same_address=1' in part
                tests[f'{mode}:stale_access_faults_25'] = mode_causes == [25] and status == '139'
                tests[f'{mode}:stale_access_did_not_read'] = 'ROOT_STALE_READ' not in part
            else:
                tests[f'{mode}:exit_zero'] = mode_causes == [] and status == '0'
            if mode == 'refuse' and a.root_grants:
                collections = line(r'^CAPSTONE_VM_STATS .*\bcollections=(\d+)', part)
                tests['refuse:collected'] = collections is not None and int(collections) >= 1
            if a.root_grants:
                victim_reason = None
                if mode == 'exit-victim':
                    operand = line(r'^ROOT_OPERAND own_stack_at_victim type=(\d+) adjacent=1 tries=\d+$', part)
                    victim_reason = 'not a reference' if operand == '7' else 'not the mapping'
                    tests['exit-victim:operand_is_a_reference'] = operand == '1'
                    stack_refusals = len(re.findall(r'^CAPSTONE_VM_STACK_UNMAP_REFUSED ', part, re.M))
                    tests['exit-victim:launcher_keeps_both_victim_unmaps'] = stack_refusals == 2
                expected = expected_refusals(mode, a.control, a.launcher_bypass, victim_reason)
                tests[f'{mode}:kernel_refusals'] = refusals_match(seen, expected)
            else:
                tests[f'{mode}:kernel_refuses_nothing'] = seen == []
        refused_count = line(r'^ROOT_UNMAP_REFUSED:(\w+)$')
        tests['kernel_count_matches_reports'] = refused_count == str(len(reports))
        tests['module_unloaded'] = 'ROOT_CLEANUP:0' in log
        tests['no_kernel_fault'] = 'Oops' not in log and 'Kernel panic' not in log

    result = dict(status='PASS' if all(tests.values()) else 'FAIL', root_grants=a.root_grants,
                  control=a.control, launcher=str(a.launcher) if a.launcher else None,
                  launcher_bypass=a.launcher_bypass, no_refs=a.no_refs, unpinned=a.unpinned,
                  passed=sum(tests.values()), total=len(tests), tests=tests,
                  insmod=insmod, exits=exits, fault_causes=causes, refusals=refusals_seen,
                  retire_refusals=retires_seen if not a.no_refs else None,
                  retire_refused=line(r'^ROOT_RETIRE_REFUSED:(\w+)$'), no_refs_outcome=outcome)
    (a.work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'passed', 'total', 'exits', 'fault_causes')}))
    for name, ok in tests.items():
        if not ok:
            print('FAIL', name)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
