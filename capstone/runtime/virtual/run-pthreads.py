#!/usr/bin/env python3
"""Qualify musl pthread lifetimes, TLS, synchronization and independent I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('adapter', 'application', 'qemu', 'images', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--exact-bounds', action=argparse.BooleanOptionalAction, default=False,
                   help="Run QEMU's exact-bounds profile. Option (iii) does not need it; "
                        "it is kept so a suite can be run both ways as a two-sided control.")
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work/'stage'
    stage.mkdir()
    paths = dict(launcher=a.adapter/'capstone-vexec', module=a.adapter/'module/capstone_vm.ko',
                 application=a.application, qemu=a.qemu, source=HERE/'pthread-contract.c')
    paths.update({name: a.images/name for name in ('Image', 'fw_jump.elf', 'rootfs.ext2')})
    for key, name in [('launcher', 'capstone-vexec'), ('module', 'capstone_vm.ko'), ('application', 'pthread.dom')]:
        shutil.copy2(paths[key], stage/name)
    # A module with the diag_admit parameter prints one line per page it admits (dmesg); the
    # exit-word check below reads the `touch` line of the page the contract names. A module
    # without it (the lazy-grants change's control) runs the gate as before and the check is
    # recorded as uninstrumented rather than passed.
    instrumented = b'diag_admit' in paths['module'].read_bytes()
    # /proc/vmstat's FOLL_PIN counters, before insmod and after rmmod (when no context is left):
    # every pin taken during the test must have been released by then. The module's census and
    # counters cannot show that -- a module that sweeps, decrements and prints but never unpins
    # keeps both consistent. The counters are guest-wide, so the balance is taken over the test
    # alone (acquired minus released between the two readings), which tolerates a pin some other
    # guest process holds throughout; it assumes no other process takes or drops pins during the
    # test without balancing them by the second reading (the staged guest runs nothing else).
    gate = '''#!/bin/sh
cd /mnt/vm || exit 1
pins() { echo 1 > /proc/sys/vm/stat_refresh; grep -E "^nr_foll_pin_(acquired|released) " /proc/vmstat | tr "\\n" " "; }
dmesg -n 1
echo "PTHREAD_PINS_BEFORE:$(pins)"
insmod capstone_vm.ko ''' + ('diag_admit=1' if instrumented else '') + ''' || exit 1
CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_STATS=1 CAPSTONE_VM_STATS_MARK=1 CAPSTONE_VM_GRANTS=1 ./capstone-vexec pthread.dom
echo PTHREAD_EXIT:$?
dmesg | grep -E "capstone_vm: |CAPSTONE_VM_STATS_MARK" | sed "s/^/PTHREAD_DIAG:/"
echo PTHREAD_FALLBACK_BEGIN
echo 0 > /sys/module/capstone_vm/parameters/diag_admit 2>/dev/null || true
CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_STATS=1 ./capstone-vexec pthread.dom fallback
echo PTHREAD_FALLBACK_EXIT:$?
rmmod capstone_vm
echo PTHREAD_CLEANUP:$?
echo "PTHREAD_PINS_END:$(pins)"
echo VIRTUAL_STAGED_DONE
'''
    (stage/'gate.sh').write_text(gate)
    run = subprocess.run([sys.executable, str(HERE/'run-staged.py'), '--qemu', str(a.qemu.resolve()),
                          '--images', str(a.images.resolve()), '--stage', str(stage),
                          '--work', str(a.work/'guest'), '--timeout', str(a.timeout)] + (['--exact-bounds'] if a.exact_bounds else ['--no-exact-bounds']))
    log = (a.work/'guest/serial.log').read_text(errors='replace').replace('\r', '')
    main_log, _, fallback_log = log.partition('PTHREAD_FALLBACK_BEGIN\n')
    checks = {name: log.count('VIRTUAL_PTHREAD_OK ' + name) == 1 for name in
              ('unsupported_clone_refused', 'tls mutex cond tagged_join', 'timed_wait', 'independent_blocking_io', 'private_epoll_events', 'shared_tls_transport', '64_joined_lifetimes', 'concurrent_allocator_remap', 'transport_lifetime', 'transport_identity', 'exit_word_untouched_stack', 'context_identity')}
    # The exit word's page (VIRTUAL_PTHREAD_WORD, in the main stack: the reg 2 grant) must have
    # been admitted by the launcher's CV_TOUCH before its store: a `touch` line for that arena and
    # page index. Without CV_TOUCH (the commit before it) the line is absent and this fails.
    word = re.search(r'^VIRTUAL_PTHREAD_WORD:0x([0-9a-f]+)$', log, re.M)
    stack = re.search(r'CAPSTONE_VM_ADD_OK slot=\d+ base=0x([0-9a-f]+) bytes=(\d+) reg=2 ', log)
    touched = set()
    for m in re.finditer(r'capstone_vm: touch arena 0x([0-9a-f]+)\+0x([0-9a-f]+) page (\d+)', log):
        touched.add((int(m.group(1), 16), int(m.group(3))))
    if not instrumented:
        checks['exit_word_admitted'] = 'uninstrumented'
    elif word and stack and int(word.group(1), 16) - int(stack.group(1), 16) < int(stack.group(2)):
        index = (int(word.group(1), 16) - int(stack.group(1), 16)) // 4096
        checks['exit_word_admitted'] = (int(stack.group(1), 16), index) in touched
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
        len(fallback_steps) == 1 and int(fallback_steps[0]) >= 960)
    checks['exit'] = log.count('PTHREAD_EXIT:0') == 1
    checks['cleanup'] = run.returncode == 0 and log.count('PTHREAD_CLEANUP:0') == 1
    record = dict(status='PASS' if all(v is True or v == 'uninstrumented' for v in checks.values()) else 'FAIL',
                  tests=checks, instrumented=instrumented,
                  harts=1, virtual_vm_abi=5 if a.exact_bounds else 3, input_sha256={name: sha(path) for name, path in paths.items()},
                  gate_sha256=hashlib.sha256(gate.encode()).hexdigest(), runner_sha256=sha(Path(__file__)))
    (a.work/'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
