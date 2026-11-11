#!/usr/bin/env python3
"""Qualify musl pthread lifetimes, TLS, synchronization and independent I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import memory_mode
import native_platform

HERE = Path(__file__).resolve().parent
SLOTS_PROPERTY = 'x-capstone-supervisor-slots='


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def staged_environment(supervisor_slots):
    """The environment for run-staged.py, which reads the slot count of QEMU and of the kernel from
    CAPSTONE_QEMU_CPU_EXTRA. An explicit count replaces the one the caller's variable names."""
    environment = dict(os.environ)
    if supervisor_slots is None:
        return environment

    words = [word.strip() for word in environment.get('CAPSTONE_QEMU_CPU_EXTRA', '').split(',')]
    kept = [word for word in words if word and not word.startswith(SLOTS_PROPERTY)]
    environment['CAPSTONE_QEMU_CPU_EXTRA'] = ','.join([*kept, f'{SLOTS_PROPERTY}{supervisor_slots}'])
    return environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('adapter', 'application', 'qemu', 'images', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--supervisor-slots', type=int, help='explicit context capacity of QEMU and the kernel for multi-thread tests')
    p.add_argument('--unpinned', action='store_true', help='require unpinned CV_TOUCH and cleanup witnesses (the native adapter has no other mode)')
    p.add_argument('--exact-bounds', action=argparse.BooleanOptionalAction, default=False,
                   help="Run QEMU's exact-bounds profile. Option (iii) does not need it; "
                        "it is kept so a suite can be run both ways as a two-sided control.")
    a = p.parse_args()
    if a.supervisor_slots is not None and a.supervisor_slots < 1:
        p.error('--supervisor-slots must be positive')
    if a.application.read_bytes().count(b'CPONVVM7') != 1:
        p.error('application has missing or ambiguous virtual ABI 7 magic')
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work/'stage'
    stage.mkdir()
    paths = dict(launcher=a.adapter/native_platform.LAUNCHER, application=a.application, qemu=a.qemu,
                 source=HERE/'pthread-contract.c')
    paths.update({name: a.images/name for name in ('Image', 'fw_jump.elf', 'rootfs.ext2')})
    for key, name in [('launcher', native_platform.LAUNCHER), ('application', 'pthread.dom')]:
        shutil.copy2(paths[key], stage/name)
    # The kernel boots with capstone_vm.diag_admit=1, and the adapter then prints one line per page
    # it admits (dmesg); the exit-word check below reads the `touch` line of the page the contract
    # names. The gate echoes the parameter. A kernel that does not report it on runs the gate all
    # the same, and the check is recorded as uninstrumented rather than passed.
    #
    # /proc/vmstat's FOLL_PIN counters, before the first image and after the last (when no context
    # is left): every pin taken during the test must have been released by then. The adapter's
    # counters cannot show that -- an adapter that sweeps, decrements and prints but never unpins
    # keeps them consistent. The counters are guest-wide, so the balance is taken over the test
    # alone (acquired minus released between the two readings), which tolerates a pin some other
    # guest process holds throughout; it assumes no other process takes or drops pins during the
    # test without balancing them by the second reading (the staged guest runs nothing else).
    gate = '''#!/bin/sh
cd /mnt/vm || exit 1
pins() { echo 1 > /proc/sys/vm/stat_refresh; grep -E "^nr_foll_pin_(acquired|released) " /proc/vmstat | tr "\\n" " "; }
dmesg -n 1
echo "PTHREAD_PINS_BEFORE:$(pins)"
''' + native_platform.admit() + '''
''' + memory_mode.echo('PTHREAD_') + '''
echo "PTHREAD_DIAG_ADMIT:$(cat /sys/module/capstone_vm/parameters/diag_admit 2>/dev/null || echo A)"
CAPSTONE_EXEC_DIAGNOSTICS=1 ./capstone-vexec pthread.dom
echo PTHREAD_EXIT:$?
dmesg | grep -E "capstone_vm: " | sed "s/^/PTHREAD_DIAG:/"
echo PTHREAD_FALLBACK_BEGIN
echo 0 > /sys/module/capstone_vm/parameters/diag_admit 2>/dev/null || true
CAPSTONE_EXEC_DIAGNOSTICS=1 ./capstone-vexec pthread.dom fallback
echo PTHREAD_FALLBACK_EXIT:$?
for mode in identity-exit identity-jump private-tls; do
  echo PTHREAD_IDENTITY_BEGIN:$mode
  ./capstone-vexec pthread.dom "$mode"
  echo PTHREAD_IDENTITY_EXIT:$mode:$?
done
''' + native_platform.leave('PTHREAD_') + '''
echo "PTHREAD_PINS_END:$(pins)"
echo VIRTUAL_STAGED_DONE
'''
    (stage/'gate.sh').write_text(gate)
    run = subprocess.run([sys.executable, str(HERE/'run-staged.py'), '--qemu', str(a.qemu.resolve()),
                          '--images', str(a.images.resolve()), '--stage', str(stage),
                          '--work', str(a.work/'guest'), '--timeout', str(a.timeout),
                          '--kernel-arg', native_platform.WIDE_GRANTS, '--kernel-arg', 'capstone_vm.diag_admit=1'
                          ] + (['--exact-bounds'] if a.exact_bounds else ['--no-exact-bounds']),
                         env=staged_environment(a.supervisor_slots))
    log = (a.work/'guest/serial.log').read_text(errors='replace').replace('\r', '')
    instrumented = log.count('PTHREAD_DIAG_ADMIT:Y\n') == 1
    main_log, _, fallback_log = log.partition('PTHREAD_FALLBACK_BEGIN\n')
    checks = {name: log.count('VIRTUAL_PTHREAD_OK ' + name) == 1 for name in
              ('unsupported_clone_refused', 'protected_fork_refused', 'process_creation_refused_or_delegated', 'tls mutex cond tagged_join', 'timed_wait', 'independent_blocking_io', 'private_epoll_events', 'shared_tls_transport', '64_joined_lifetimes', 'concurrent_allocator_remap', 'transport_lifetime', 'transport_identity', 'exit_word_untouched_stack', 'context_identity')}
    checks['memory_mode_verified'] = memory_mode.verified(log, a.unpinned, 1)
    # The exit word's page (VIRTUAL_PTHREAD_WORD, in the main stack) must have been admitted by
    # CV_TOUCH before its store: a `touch` line for the arena that holds the word, at the word's
    # page index. The line names the arena's base and length, so the word itself selects the
    # arena. Without CV_TOUCH the line is absent and this fails.
    word = re.search(r'^VIRTUAL_PTHREAD_WORD:0x([0-9a-f]+)$', log, re.M)
    stack = re.search(r'CAPSTONE_VM_ADD_OK slot=\d+ base=0x([0-9a-f]+) bytes=(\d+) reg=2 ', log)
    touch, where = ('unpinned touch', main_log) if a.unpinned else ('touch', log)
    touched = {(int(m[1], 16), int(m[2], 16), int(m[3])) for m in re.finditer(
        r'capstone_vm: ' + touch + r' arena 0x([0-9a-f]+)\+0x([0-9a-f]+) page (\d+)', where)}
    if not instrumented:
        checks['exit_word_admitted'] = 'uninstrumented'
    elif word:
        address = int(word.group(1), 16)
        checks['exit_word_admitted'] = any(base <= address < base + size and page == (address - base) // 4096
                                           for base, size, page in touched)
    else:
        checks['exit_word_admitted'] = False
    # The module's own counters against its per-arena census (the retire lines, in dmesg order,
    # the launcher's CAPSTONE_VM_STATS_MARK written right before it read pages=): the stack
    # arena's sweep must cover every page it admitted, the touched page included, and pages=
    # must equal what the arenas retired after the mark still held. A module that records the
    # touch line but drops the pages[] entry fails both.
    retire = re.compile(r'capstone_vm: retire arena 0x([0-9a-f]+)\+0x[0-9a-f]+ admit=(\d+) window=(\d+) '
                        r'scan=(\d+) touch=(\d+) eager=(\d+) released=(\d+) swept=(\d+)')
    stats = re.search(r'^CAPSTONE_VM_STATS arenas=\d+ pages=(\d+) ', log, re.M)
    if not instrumented:
        checks['stack_sweep_complete'] = checks['pages_match_census'] = 'uninstrumented'
    else:
        after_mark, alive, sweep_ok, seen_stack = False, 0, None, False
        for line in log.splitlines():
            if 'CAPSTONE_VM_STATS_MARK' in line:
                after_mark = True
                continue
            m = retire.search(line)
            if not m:
                continue
            base, admit, window, scan, touch, eager, released, swept = (int(m.group(1), 16),) + tuple(
                int(m.group(i)) for i in range(2, 9))
            held = admit + window + scan + touch + eager - released
            if after_mark:
                alive += held
            if stack and base == int(stack.group(1), 16):
                seen_stack = True
                sweep_ok = (swept == held and touch >= 1)
        checks['stack_sweep_complete'] = bool(seen_stack and sweep_ok)
        checks['pages_match_census'] = bool(stats and after_mark and int(stats.group(1)) == alive)
    if a.unpinned:
        checks['exit_word_resolved'] = checks.pop('exit_word_admitted') is True
        del checks['stack_sweep_complete']
        raw = re.findall(r'unpinned retire arena 0x([0-9a-f]+) events=(\d+) resolves=(\d+) touch=(\d+)', main_log)
        checks['stack_unpinned_retired'] = bool(stack and stats and int(stats[1]) == 0 and
            any(int(base, 16) == int(stack[1], 16) and int(touch) > 0 for base, events, resolves, touch in raw))
    def pins(tag):
        m = re.search('^' + tag + r':(.*)$', log, re.M)
        v = dict(re.findall(r'nr_foll_pin_(acquired|released) (\d+)', m.group(1))) if m else {}
        return (int(v['acquired']), int(v['released'])) if len(v) == 2 else None
    before, end = pins('PTHREAD_PINS_BEFORE'), pins('PTHREAD_PINS_END')
    checks['pins_balanced'] = bool(before and end and end[0] - before[0] == end[1] - before[1]
                                   and end[0] > before[0])
    # The transport lock reads each context's identity from its gp (delegate.c dl_wire_identity):
    # every context of the contract has an identity slot, so none asks the launcher for one.
    # The launcher counts the service per run (CAPSTONE_VM_STEPS, CV_SERVICE_THREAD_SELF = 11).
    steps = re.findall(r'^CAPSTONE_VM_STEPS .*\bservice11_thread_self=(\d+)', main_log, re.M)
    checks['no_thread_self_exits'] = len(steps) == 1 and steps[0] == '0'
    fallback_steps = re.findall(r'^CAPSTONE_VM_STEPS .*\bservice11_thread_self=(\d+)', fallback_log, re.M)
    checks['fallback_identity'] = (
        fallback_log.count('VIRTUAL_PTHREAD_OK fallback_identity') == 1 and
        fallback_log.count('PTHREAD_FALLBACK_EXIT:0') == 1 and
        len(fallback_steps) == 1 and int(fallback_steps[0]) >= 256)
    checks['fallback_pthread'] = (
        fallback_log.count('VIRTUAL_PTHREAD_OK fallback_pthread') == 1 and
        fallback_log.count('VIRTUAL_FALLBACK_PTHREAD:slots=4 records=128 bad=0') == 1 and
        fallback_log.count('PTHREAD_FALLBACK_EXIT:0') == 1)
    for mode in ('identity-exit', 'identity-jump'):
        part = re.search(r'PTHREAD_IDENTITY_BEGIN:' + mode + r'\n(.*?)PTHREAD_IDENTITY_EXIT:' +
                         mode + r':0\n', log, re.S)
        checks[mode] = bool(part and 'VIRTUAL_PTHREAD_OK abandoned_identity\n' in part[1])
    part = re.search(r'PTHREAD_IDENTITY_BEGIN:private-tls\n(.*?)PTHREAD_IDENTITY_EXIT:private-tls:0\n', log, re.S)
    checks['raw_create_refused'] = 'VIRTUAL_PTHREAD_OK raw_create_refused\n' in main_log
    checks['private_tls_refused'] = bool(part and 'VIRTUAL_PTHREAD_OK private_tls_refused\n' in part[1])
    checks['private_tls_other_refused'] = bool(part and 'VIRTUAL_PTHREAD_OK private_tls_other_refused\n' in part[1])
    checks['exit'] = log.count('PTHREAD_EXIT:0') == 1
    checks['cleanup'] = run.returncode == 0 and log.count('PTHREAD_CLEANUP:0') == 1
    # A native task runs without the launcher's loop, so nothing prints the launcher's grant lines
    # (CAPSTONE_VM_ADD_OK), its exit statistics (CAPSTONE_VM_STATS, after CAPSTONE_VM_STATS_MARK) or
    # its step counts (CAPSTONE_VM_STEPS). A check that reads a line its part of the log does not
    # hold leaves the verdict, and the record names it as not observable.
    reads = {'pages_match_census': (log, 'CAPSTONE_VM_STATS ', 'CAPSTONE_VM_STATS_MARK'),
             'stack_sweep_complete': (log, 'CAPSTONE_VM_ADD_OK '),
             'stack_unpinned_retired': (log, 'CAPSTONE_VM_ADD_OK ', 'CAPSTONE_VM_STATS '),
             'no_thread_self_exits': (main_log, 'CAPSTONE_VM_STEPS '),
             'fallback_identity': (fallback_log, 'CAPSTONE_VM_STEPS ')}
    not_observable = [name for name, (part, *lines) in reads.items()
                      if name in checks and not all(line in part for line in lines)]
    for name in not_observable:
        del checks[name]
    record = dict(status='PASS' if all(v is True or v == 'uninstrumented' for v in checks.values()) else 'FAIL',
                  tests=checks, not_observable=not_observable, instrumented=instrumented,
                  harts=1, virtual_vm_abi=7, exact_bounds=a.exact_bounds, supervisor_slots=a.supervisor_slots, input_sha256={name: sha(path) for name, path in paths.items()},
                  gate_sha256=hashlib.sha256(gate.encode()).hexdigest(), runner_sha256=sha(Path(__file__)))
    (a.work/'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    for name in not_observable:
        print('NOT OBSERVABLE ' + name)
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
