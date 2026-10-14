#!/usr/bin/env python3
"""CV_STEP's stop checks and the pin-ahead window, under injected conditions.

Needs the TEST build of the adapter module (make CV_TEST_HOOKS=1 in
module/), passed with --module; the production module has none of the
hooks. A hook injects need_resched, a signal (TIF_NOTIFY_SIGNAL), an expired
deadline or SIGKILL at one point of CV_STEP: right after a demand GUP (1), in
the interrupt-enabled window before a later CSRUNV (2), or before the first
CSRUNV of an ioctl (3). The module counts injections, the stops they caused
and any CSRUNV entered while one was pending (a violation), plus every GUP
request and its size, every stop check made with interrupts on, and every
CSRUNV entry, and prints them at rmmod. /proc/vmstat's FOLL_PIN counters show
that every pin taken was released, including over the pin-ahead modes,
retirement, a dying domain and, with --malloc-application (malloc.dom from
malloc-contract.c), mremap growth, shrinking and failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
# name: (insmod parameters, modes, expected exit status[, application])
CONFIGS = {
    'base1':      ('pin_ahead_pages=1', 'walk syscalls', 0),
    'base16':     ('pin_ahead_pages=16', 'walk', 0),
    'base64':     ('pin_ahead_pages=64', 'walk seq', 0),
    'resched2':   ('test_inject=1 test_inject_at=2', 'walk syscalls', 0),
    'signal2':    ('test_inject=2 test_inject_at=2', 'walk syscalls', 0),
    'deadline2':  ('test_inject=3 test_inject_at=2', 'walk syscalls', 0),
    'resched1':   ('test_inject=1 test_inject_at=1 pin_ahead_pages=16', 'walk', 0),
    'signal1':    ('test_inject=2 test_inject_at=1 pin_ahead_pages=16', 'walk', 0),
    'deadline1':  ('test_inject=3 test_inject_at=1 pin_ahead_pages=16', 'walk', 0),
    'resched3':   ('test_inject=1 test_inject_at=3 test_inject_every=2', 'walk syscalls', 0),
    'signal3':    ('test_inject=2 test_inject_at=3 test_inject_every=2', 'walk syscalls', 0),
    'kill1':      ('test_inject=4 test_inject_at=1 test_inject_every=4 pin_ahead_pages=64', 'walk', 137),
    'kill2':      ('test_inject=4 test_inject_at=2 test_inject_every=4 pin_ahead_pages=16', 'walk', 137),
    'short64':    ('test_short_gup=1 pin_ahead_pages=64', 'walk seq', 0),
    'life16':     ('pin_ahead_pages=16', 'seq stop vma vmax retire', 0),
    'life64':     ('pin_ahead_pages=64', 'seq stop vma vmax retire', 0),
    'ro64':       ('pin_ahead_pages=64', 'rowrite', 139),
    'malloc16':   ('pin_ahead_pages=16', 'ok', 0, 'malloc'),
    'malloc64':   ('pin_ahead_pages=64', 'ok', 0, 'malloc'),
}
SCRIPT = '''#!/bin/sh
cd /mnt/vm || exit 1
export CAPSTONE_VM_STATS=1
pins() { echo 1 > /proc/sys/vm/stat_refresh; grep -E "^nr_foll_pin_(acquired|released) " /proc/vmstat | tr "\\n" " "; }
'''
def app(name):
    return CONFIGS[name][3] if len(CONFIGS[name]) > 3 else 'pin'


def script(names):
    text = SCRIPT
    for name in names:
        params, modes = CONFIGS[name][:2]
        text += f'''insmod capstone_vm.ko {params} || exit 1
for mode in {modes}; do
  echo "HOOK_PINS_BEFORE:{name}:$mode:$(pins)"
  ./capstone-vexec {app(name)}.dom "$mode" > /tmp/pin.out 2>&1
  echo "HOOK_EXIT:{name}:$mode:$?"
  echo "HOOK_PINS_AFTER:{name}:$mode:$(pins)"
  cat /tmp/pin.out
done
./capstone-vexec --stats
rmmod capstone_vm || exit 1
echo "HOOK_PINS_END:{name}:$(pins)"
dmesg | grep "capstone_vm test unload" | tail -1 | sed "s/^.*capstone_vm test unload/HOOK_REPORT:{name}/"
'''
    return text + 'echo VIRTUAL_STAGED_DONE\n'


def pins(text):
    v = dict(re.findall(r'nr_foll_pin_(acquired|released) (\d+)', text))
    return int(v['acquired']), int(v['released'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qemu', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True, help='capstone-vexec from here')
    p.add_argument('--module', type=Path, required=True, help='the CV_TEST_HOOKS build')
    p.add_argument('--application', type=Path, required=True, help='pin.dom')
    p.add_argument('--malloc-application', type=Path, help='malloc.dom; without it the malloc configurations are skipped')
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'; stage.mkdir()
    names = [n for n in CONFIGS if app(n) == 'pin' or a.malloc_application]
    inputs = {'qemu': a.qemu, 'application': a.application,
              'launcher': a.adapter/'capstone-vexec', 'module': a.module,
              **{n: a.images/n for n in ('Image', 'fw_jump.elf', 'rootfs.ext2')}}
    if a.malloc_application:
        inputs['malloc'] = a.malloc_application
    hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    if a.malloc_application:
        shutil.copy2(a.malloc_application, stage/'malloc.dom')
    for src, name in ((inputs['application'], 'pin.dom'),
                      (inputs['launcher'], 'capstone-vexec'),
                      (inputs['module'], 'capstone_vm.ko')):
        shutil.copy2(src, stage/name)
    gate = script(names)
    (stage/'gate.sh').write_text(gate)
    run = subprocess.run([os.environ.get('PYTHON', 'python3'),
        str(HERE/'run-staged.py'), '--qemu', str(a.qemu), '--images', str(a.images),
        '--stage', str(stage), '--work', str(a.work/'guest'), '--disk-mib', '64',
        '--timeout', str(a.timeout)])
    log = (a.work/'guest/serial.log').read_text(errors='replace').replace('\r', '')
    runs, reports, ends, current = {}, {}, {}, None
    for line in log.splitlines():
        m = re.match(r'HOOK_PINS_BEFORE:(\w+):(\w+):(.*)$', line)
        if m:
            current = (m.group(1), m.group(2))
            runs[current] = {'before': pins(m.group(3)), 'stats': None, 'ok': False}
            continue
        m = re.match(r'HOOK_EXIT:(\w+):(\w+):(\d+)$', line)
        if m and current == (m.group(1), m.group(2)):
            runs[current]['exit'] = int(m.group(3)); continue
        m = re.match(r'HOOK_PINS_AFTER:(\w+):(\w+):(.*)$', line)
        if m and current == (m.group(1), m.group(2)):
            runs[current]['after'] = pins(m.group(3)); continue
        m = re.match(r'HOOK_PINS_END:(\w+):(.*)$', line)
        if m:
            ends[m.group(1)] = pins(m.group(2)); continue
        m = re.match(r'HOOK_REPORT:(\w+): (.*)$', line)
        if m:
            reports[m.group(1)] = dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', m.group(2)))
            continue
        if current and line.startswith('CAPSTONE_VM_STATS '):
            runs[current]['stats'] = dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
        elif current and (line == 'PIN_OK:' + current[1] or
                          (app(current[0]) == 'malloc' and line == 'MALLOC_TAG_COPY_OK')):
            runs[current]['ok'] = True
    census = [json.loads(l) for l in log.splitlines() if l.startswith('{"version":1,')]
    expected_runs = sum(len(CONFIGS[n][1].split()) for n in names)
    checks = {'completed': run.returncode == 0, 'all_runs': len(runs) == expected_runs,
              'all_reports': sorted(reports) == sorted(names),
              'no_contract_failure': 'PIN_FAIL:' not in log and 'MALLOC_FAIL:' not in log,
              'cleanup': len(census) == len(names) and all(
                  s['live_domains'] == s['live_regions'] == s['live_bytes'] == 0 for s in census)}

    # Pinned pages and fault events (CSRUNV entries also count quantum
    # preemptions, which depend on timing).
    def counts(name, mode):
        s = runs.get((name, mode), {}).get('stats')
        return s and (s['pages'], s['faults'])

    for (name, mode), r in runs.items():
        exit_status = CONFIGS[name][2]
        checks[f'{name}_{mode}_exit'] = r.get('exit') == exit_status and r['ok'] == (exit_status == 0)
    # Releases can trail a process's exit (the adapter fd closes as it
    # dies), so the balance is taken after rmmod, when no context is left:
    # every pin Linux recorded as acquired has been released.
    for name in names:
        e = ends.get(name)
        checks[f'{name}_pins_balanced'] = bool(e and e[0] == e[1] and e[0] > 0)
    for name, rep in reports.items():
        params = CONFIGS[name][0]
        if 'test_inject=' in params:
            checks[f'{name}_injected'] = rep['injections'] > 0
            # A need_resched consumed by a reschedule inside GUP needs no stop.
            checks[f'{name}_every_injection_stopped'] = (
                rep['stops'] + rep['consumed'] == rep['injections'] and
                (rep['consumed'] == 0 or 'test_inject=1 ' in params + ' '))
            checks[f'{name}_no_entry_while_pending'] = rep['violations'] == 0
        if 'test_inject_at=3' in params:
            checks[f'{name}_initial_stops'] = rep['initial_stops'] == rep['injections']
        checks[f'{name}_gup_batch_bound'] = 0 < rep['gup_max'] <= 16
        # Every stop check ran with interrupts off, right before CSRUNV.
        checks[f'{name}_checks_irqs_off'] = rep['checks'] > 0 and rep['checks_irqs_on'] == 0
        # The launcher's CSRUNV-entry count is the module's own (a stopped
        # first entry is no entry); a killed process prints no stats.
        if CONFIGS[name][2] == 0:
            steps = sum(r['stats']['steps'] for (n, m), r in runs.items()
                        if n == name and r['stats'])
            checks[f'{name}_steps_are_entries'] = steps == rep['entries']
    checks['base64_full_batches'] = reports.get('base64', {}).get('gup_max') == 16
    checks['short64_partial_results'] = reports.get('short64', {}).get('short_results', 0) > 0
    # A stop changes no accounting: same pins and fault events as the
    # uninjected run with the same effective window.
    for name, ref in (('resched2', 'base1'), ('signal2', 'base1'), ('deadline2', 'base1'),
                      ('resched3', 'base1'), ('signal3', 'base1'),
                      ('resched1', 'base16'), ('signal1', 'base1'), ('deadline1', 'base1')):
        for mode in CONFIGS[name][1].split():
            got, want = counts(name, mode), counts(ref, mode)
            checks[f'{name}_{mode}_accounting'] = bool(got and want and got == want)
    # Partial GUP results leave the rest to later faults: every page the walk
    # touched still ends pinned (at least window 1's pins), after more faults
    # than the full window takes.
    got, low, full = counts('short64', 'walk'), counts('base1', 'walk'), counts('base64', 'walk')
    checks['short64_walk_pins'] = bool(got and low and full and got[0] >= low[0] and got[1] > full[1])
    checks['inputs_unchanged'] = hashes == {
        k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    result = {'schema': 1, 'status': 'PASS' if all(checks.values()) else 'FAIL',
              'checks': checks, 'reports': reports, 'pins_at_rmmod': ends,
              'runs': {f'{n}:{m}': v for (n, m), v in runs.items()}, 'sha256': hashes,
              'source_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), HERE/'pin-ahead-contract.c',
                                          HERE/'run-staged.py', HERE/'module/capstone_vm.c']},
              'gate_sha256': hashlib.sha256(gate.encode()).hexdigest()}
    (a.work/'result.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, ok in checks.items(): print(('PASS ' if ok else 'FAIL ') + name)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
