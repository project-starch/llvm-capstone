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
  mremap       the libc's map-copy-unmap mremap: refused with a wrong old length and through an
               object, then grows and shrinks with data and a stored reference; ends with a fault
               through the old reference (139)
  exit-victim  a thread exits naming a victim mapping it holds no reference to (exit 0)
  exit-renew   a thread renews its stack's mapping and exits; the main thread's copy of the renewed
               reference must fault with 25 (139)
  create       the frame's hand-over with the reference from before CRENEW (refused) and with the
               renewed one; the renewed reference must fault with 25 afterwards (139)
  create-fail  the launcher fails a create after the module took the frame: the frame's mapping is
               released (exit 0)
  detached     a detached pthread's stack is unmapped at its exit (exit 0)
  exit-length  a thread exits naming its stack with a wrong length: the exit completes, the stack
               stays mapped and unmaps later through the kept reference (exit 0)
  exit-huge    the same with the length 2^64 - 1 and an exit word outside the stack (exit 0)
  exit-frame   a thread exits naming its own frame as its stack: the module keeps the stack and
               releases the frame once (exit 0)
  protect      mprotect through the mapping's reference, part of it, and refused through an integer
               address, a stale or narrowed reference and a nested object (exit 0)
  protect-held a pthread's mprotect of its own exchange region, held by the launcher: EBUSY; after
               the thread detaches its transport, the same mprotect succeeds (exit 0)
  create-gate  thread create with a metadata page, the launcher's stack and a frame reference
               moved inside the page: refused; then through the frame's own reference (exit 0)
Every refusal of a request is counted (parameter unmap_refused) and reported in the kernel log
with its reason and request, and so is every CV_RETIRE that the module refuses because the
mapping's reservation is live (parameter retire_refused; no launcher path asks for one any more)
and every arena that a thread exit names but keeps (parameter exit_kept). The runner reads them
back per mode (the log is read and cleared after each mode) and compares them with what each mode
does. Every thread's frame must be released at its exit (no CAPSTONE_VM_FRAME_RELEASE_FAILED).

--root-grants 0 loads the module without the parameter. The checks that hold only with root
grants print ROOT_SKIP with a reason, and the runner requires those skips instead.
--control NAME runs against a module that is one change away from the adapter's (--module):
  cover          the unmap check skips condition 4 (the bounds cover the mapping): the narrowed
                 munmap is accepted, and the narrow mode fails at that check (exit 1)
  parent         the check skips condition 3 (the operand's record is a child of M): the object
                 over the whole page ends the whole mapping, and the refuse mode fails there; the
                 protect mode fails at the object's mprotect and the mremap mode at the object's
                 mremap
  exit-operand   the stack's arena at thread exit is released without the CREVOKE of M: the
                 renewed stack lifetime survives, and exit-renew reads through it
  protect-check  CV_PROTECT accepts without checking the reference: the protect mode fails at the
                 integer address (exit 1)
  create-address thread create without the comparison of the launcher's frame with a0's cursor:
                 with --launcher-frame-floor the create-gate mode fails at the moved reference
Every other mode must pass as without the control.
--launcher PATH runs another capstone-vexec. --launcher-bypass says that it lacks the launcher's
whole-mapping check: the partial munmap, the mremap with a wrong old length and the stacks that
exit-length and exit-huge name with a wrong length then reach the module, which must refuse each
(reason "range is not the mapping") and complete the exits. --launcher-frame-floor says that it
names the mapping around a0's cursor as the frame at thread create: the moved frame reference of
create-gate then reaches the module, which must refuse it (reason "range is not the mapping").
--launcher-control NAME says that the --launcher is one change away from the adapter's:
  old-exit-word  the exit word is checked against the stack base and length that the program
                 names, as before stage 4: exit-huge stops the launcher at the exit service
  no-hold        protect_range() lacks the check of the transport hold: protect-held fails at the
                 held mapping's mprotect (exit 1)
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
         'exit-victim', 'exit-renew', 'create', 'create-fail', 'detached',
         'exit-length', 'exit-huge', 'exit-frame', 'protect', 'protect-held', 'create-gate')
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
    'mremap': ('mremap_wrong_length_refused', 'mremap_nested_object_refused', 'public_mremap_grows',
               'mremap_old_reference_refused', 'public_mremap_shrinks'),
    'exit-victim': ('exit_with_own_stack_at_victim_refused', 'exit_with_victim_address_refused',
                    'victim_unmaps_afterwards'),
    'exit-renew': ('stack_renewed_before_exit',),
    'create': ('create_with_stale_frame_refused', 'create_with_renewed_frame'),
    'create-fail': ('failed_create_releases_frame',),
    'detached': ('detached_exit_unmaps_stack',),
    'exit-length': ('exit_with_wrong_length_completes', 'wrong_length_stack_stays_mapped',
                    'wrong_length_stack_unmaps_afterwards'),
    'exit-huge': ('exit_with_huge_length_completes', 'huge_length_stack_stays_mapped',
                  'huge_length_stack_unmaps_afterwards'),
    'exit-frame': ('exit_with_frame_as_stack_completes', 'frame_as_stack_released_once'),
    'protect': ('protect_through_mapping_reference', 'protect_integer_address_refused',
                'protect_stale_reference_refused', 'protect_narrowed_reference_refused',
                'protect_nested_object_refused', 'protect_narrowed_covering_range'),
    'protect-held': ('protect_held_mapping_refused', 'protect_after_detach'),
    'create-gate': ('create_metadata_frame_refused', 'create_launcher_page_refused',
                    'create_mismatched_frame_refused', 'create_after_refusals'),
}
ROOT_ONLY = {'stale_reference_at_reused_address_refused', 'address_refused',
             'nested_object_refused', 'stale_reference_refused',
             'stale_reference_after_collection_refused', 'narrowed_reference_refused',
             'whole_reference_after_narrowed', 'renewed_old_reference_refused',
             'exit_with_own_stack_at_victim_refused', 'exit_with_victim_address_refused',
             'victim_unmaps_afterwards', 'create_with_stale_frame_refused',
             'protect_integer_address_refused', 'protect_stale_reference_refused',
             'protect_narrowed_reference_refused', 'protect_nested_object_refused',
             'mremap_nested_object_refused'}
FAULTING = {'unmap', 'renew', 'renew-ref', 'mremap', 'exit-renew', 'create'}
# Modes whose stale read does not map the address again first (the frame is not reused).
NO_REMAP = {'create'}
# The kernel's refusals per mode with root grants, as (reason, request) in order
# (capstone_vm.c cv_unmap_reasons, request_refused). The other refusals (a second munmap, the
# second page of a partial munmap, the old reference after mremap, the wrong old length of mremap)
# are the launcher's, which finds no whole mapping at that address or of that length. exit-victim's
# first exit names the victim with the own stack's reference moved there, the stack's end: a live reference to another
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
    'mremap': [('not the mapping', 'munmap')],
    'exit-victim': [(None, 'thread exit'), ('not a reference', 'thread exit')],
    'exit-renew': [],
    'create': [('not live', 'thread create')],
    'create-fail': [],
    'detached': [],
    'exit-length': [],
    'exit-huge': [],
    'exit-frame': [],
    'protect': [('not a reference', 'mprotect'), ('not live', 'mprotect'),
                ('does not cover the mapping', 'mprotect'), ('not the mapping', 'mprotect')],
    'protect-held': [],
    'create-gate': [],
}
# What each control changes: the mode it must break, and how.
CONTROL_CUTS = {
    'cover': {'narrow': ('ROOT_FAIL:munmap through a narrowed reference', 'narrowed_reference_refused')},
    'parent': {'refuse': ('ROOT_FAIL:munmap through an object over the whole page', 'nested_object_refused'),
               'protect': ('ROOT_FAIL:mprotect through an object over the whole page',
                           'protect_nested_object_refused'),
               'mremap': ('ROOT_FAIL:mremap through an object over the whole mapping',
                          'mremap_nested_object_refused')},
    'exit-operand': {'exit-renew': ('ROOT_STALE_READ', None)},
    'protect-check': {'protect': ('ROOT_FAIL:mprotect with an integer address',
                                  'protect_integer_address_refused')},
    'create-address': {'create-gate': ('ROOT_FAIL:thread create with a frame reference whose cursor '
                                       'is not the frame\'s address', 'create_mismatched_frame_refused')},
}
# Launcher controls (--launcher-control): the mode each must break, and how.
LAUNCHER_CUTS = {
    'old-exit-word': {'exit-huge': ('capstone-vexec: service:', 'exit_with_huge_length_completes')},
    'no-hold': {'protect-held': ('ROOT_FAIL:mprotect of a held mapping', 'protect_held_mapping_refused')},
}
# The arenas that a thread exit names and the module keeps, per mode, as the reasons it reports:
# the two stacks that exit-victim's threads named (root grants only), and the frame that
# exit-frame's thread named as its stack. exit-length's and exit-huge's stacks reach the module
# only behind a launcher without its length check (--launcher-bypass); otherwise the launcher
# reports them (CAPSTONE_VM_STACK_UNMAP_REFUSED) and the module not.
EXIT_KEPT = {'exit-victim': ['request refused'] * 2, 'exit-frame': ["is the thread's frame"]}
EXIT_KEPT_RG0 = {'exit-frame': ["is the thread's frame"]}
EXIT_KEPT_BYPASS = {'exit-length': ['request refused'], 'exit-huge': ['request refused']}
# The launcher's reports of a stack it leaves mapped at a thread exit, per mode.
STACKS_KEPT = {'exit-victim': 2, 'exit-length': 1, 'exit-huge': 1, 'exit-frame': 1}
REPORT = re.compile(r'^ROOT_KERNEL:([\w-]+):unmap refused \(([^)]*)\) at ([a-z ]+):', re.M)
RETIRE_REPORT = re.compile(r'^ROOT_KERNEL:([\w-]+):retire refused:', re.M)
KEPT_REPORT = re.compile(r'^ROOT_KERNEL:([\w-]+):thread exit keeps the (stack|frame) \(([^)]*)\)', re.M)


def refusals_match(seen, expected):
    """seen matches expected entry by entry; an expected reason may be a tuple of alternatives."""
    if len(seen) != len(expected):
        return False
    for (reason, request), (want, want_request) in zip(seen, expected):
        wants = want if isinstance(want, tuple) else (want,)
        if reason not in wants or request != want_request:
            return False
    return True


def expected_refusals(mode, control, bypass, frame_floor, victim_reason):
    """The (reason, request) list a mode's kernel log must show, or None where a control cuts the
    mode short (then only a prefix is checked)."""
    refusals = list(REFUSALS[mode])
    if frame_floor and mode == 'create-gate' and control != 'create-address':
        # The frame reference moved inside the page reaches the module, which compares it.
        refusals.append(('range is not the mapping', 'thread create'))
    if control == 'protect-check' and mode == 'protect':
        return []
    if mode == 'exit-victim':
        refusals[0] = (victim_reason, 'thread exit')
    if bypass and mode == 'refuse':
        # The first page of the partial munmap reaches the module; the second page does not
        # start a mapping and the launcher still refuses it.
        refusals.insert(2, ('range is not the mapping', 'munmap'))
    if bypass and mode == 'mremap':
        # The unmap of the old mapping with the wrong old length.
        refusals.insert(0, ('range is not the mapping', 'munmap'))
    if bypass and mode in ('exit-length', 'exit-huge'):
        # The stack named with a wrong length.
        refusals.append(('range is not the mapping', 'thread exit'))
    if control == 'parent' and mode == 'refuse':
        return refusals[:1]                 # the address; the nested object is accepted
    if control == 'parent' and mode == 'protect':
        return refusals[:3]                 # the nested object is accepted
    if control == 'parent' and mode == 'mremap':
        return []                           # the nested object is accepted
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
    p.add_argument('--launcher-frame-floor', action='store_true',
                   help='the launcher names the mapping around a0\'s cursor as the frame; the module must compare')
    p.add_argument('--root-grants', type=int, choices=(0, 1), default=1)
    p.add_argument('--control', choices=sorted(CONTROL_CUTS), help='the module is this control (see above)')
    p.add_argument('--launcher-control', choices=sorted(LAUNCHER_CUTS),
                   help='the --launcher is this control (see above)')
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
    if a.launcher_control and not a.launcher:
        p.error('--launcher-control needs --launcher')
    if a.control == 'create-address' and not a.launcher_frame_floor:
        p.error('--control create-address needs a --launcher built --launcher-frame-floor')

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
dmesg -c | grep 'capstone_vm: \\(unmap refused\\|retire refused\\|thread exit keeps\\)' | sed 's/^.*capstone_vm: /ROOT_KERNEL:{mode}:/'
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
echo "ROOT_EXIT_KEPT:$(cat /sys/module/capstone_vm/parameters/exit_kept 2>/dev/null || echo A)"
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
        cuts = dict(CONTROL_CUTS.get(a.control, {}))
        cuts.update(LAUNCHER_CUTS.get(a.launcher_control, {}))
        reports = REPORT.findall(log)
        retire_reports = RETIRE_REPORT.findall(log)
        kept_reports = KEPT_REPORT.findall(log)
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
            tests[f'{mode}:kernel_retire_refusals'] = retires_seen[mode] == 0
            kept = [(what, why) for m, what, why in kept_reports if m == mode]
            if a.root_grants:
                want_kept = EXIT_KEPT.get(mode, []) + (EXIT_KEPT_BYPASS.get(mode, []) if a.launcher_bypass else [])
            else:
                want_kept = EXIT_KEPT_RG0.get(mode, [])
            tests[f'{mode}:kernel_exit_kept'] = kept == [('stack', why) for why in want_kept]
            tests[f'{mode}:frames_released'] = 'CAPSTONE_VM_FRAME_RELEASE_FAILED' not in part
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
                    expected = expected_refusals(mode, a.control, a.launcher_bypass,
                                                 a.launcher_frame_floor, None)
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
            if mode in STACKS_KEPT and (a.root_grants or mode != 'exit-victim'):
                stacks_kept = len(re.findall(r'^CAPSTONE_VM_STACK_UNMAP_REFUSED ', part, re.M))
                tests[f'{mode}:launcher_stacks_kept'] = stacks_kept == STACKS_KEPT[mode]
            if mode == 'refuse' and a.root_grants:
                collections = line(r'^CAPSTONE_VM_STATS .*\bcollections=(\d+)', part)
                tests['refuse:collected'] = collections is not None and int(collections) >= 1
            if a.root_grants:
                victim_reason = None
                if mode == 'exit-victim':
                    operand = line(r'^ROOT_OPERAND own_stack_at_victim type=(\d+) adjacent=1 tries=\d+$', part)
                    victim_reason = 'not a reference' if operand == '7' else 'not the mapping'
                    tests['exit-victim:operand_is_a_reference'] = operand == '1'
                expected = expected_refusals(mode, a.control, a.launcher_bypass,
                                             a.launcher_frame_floor, victim_reason)
                tests[f'{mode}:kernel_refusals'] = refusals_match(seen, expected)
            else:
                tests[f'{mode}:kernel_refuses_nothing'] = seen == []
        refused_count = line(r'^ROOT_UNMAP_REFUSED:(\w+)$')
        tests['kernel_count_matches_reports'] = refused_count == str(len(reports))
        tests['kernel_kept_count_matches_reports'] = line(r'^ROOT_EXIT_KEPT:(\w+)$') == str(len(kept_reports))
        tests['module_unloaded'] = 'ROOT_CLEANUP:0' in log
        tests['no_kernel_fault'] = 'Oops' not in log and 'Kernel panic' not in log

    result = dict(status='PASS' if all(tests.values()) else 'FAIL', root_grants=a.root_grants,
                  control=a.control, launcher_control=a.launcher_control,
                  launcher=str(a.launcher) if a.launcher else None,
                  launcher_bypass=a.launcher_bypass, launcher_frame_floor=a.launcher_frame_floor,
                  no_refs=a.no_refs, unpinned=a.unpinned,
                  passed=sum(tests.values()), total=len(tests), tests=tests,
                  insmod=insmod, exits=exits, fault_causes=causes, refusals=refusals_seen,
                  retire_refusals=retires_seen if not a.no_refs else None,
                  retire_refused=line(r'^ROOT_RETIRE_REFUSED:(\w+)$'),
                  exit_kept=line(r'^ROOT_EXIT_KEPT:(\w+)$'), no_refs_outcome=outcome)
    (a.work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'passed', 'total', 'exits', 'fault_causes')}))
    for name, ok in tests.items():
        if not ok:
            print('FAIL', name)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
