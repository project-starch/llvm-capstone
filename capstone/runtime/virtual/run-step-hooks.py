#!/usr/bin/env python3
"""CV_STEP's stop checks and the pin-ahead window, under injected conditions.

Needs the TEST build of the adapter module (make CV_TEST_HOOKS=1 in
module/), passed with --module; the production module has none of the
hooks. A hook injects need_resched, a signal (TIF_NOTIFY_SIGNAL), an expired
deadline or SIGKILL at one point of CV_STEP: right after a demand GUP (1), in
the interrupt-enabled window before a later CSRUNV (2), before the first
CSRUNV of an ioctl (3), or after CSRUNV returns a fault, before it is
resolved (4). The module counts injections, the stops they caused, any CSRUNV
entered and any GUP requested while one was pending (violations), plus every
GUP request and its size, every stop check made with interrupts on, and every
CSRUNV entry, and prints them at rmmod. /proc/vmstat's FOLL_PIN counters show
that every pin taken was released, including over the pin-ahead modes,
retirement, a dying domain and, with --malloc-application (malloc.dom from
malloc-contract.c), mremap growth, shrinking and failure.

--evaluate <serial.log> re-runs the checks on a saved transcript (no guest),
for checking the checks: a transcript whose CAPSTONE_VM_STATS counters are
zeroed must fail."""
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
    'resched4':   ('test_inject=1 test_inject_at=4 pin_ahead_pages=16', 'walk', 0),
    'signal4':    ('test_inject=2 test_inject_at=4 pin_ahead_pages=16', 'walk', 0),
    'deadline4':  ('test_inject=3 test_inject_at=4 pin_ahead_pages=16', 'walk', 0),
    'kill4':      ('test_inject=4 test_inject_at=4 test_inject_every=4 pin_ahead_pages=16', 'walk', 137),
    'short64':    ('test_short_gup=1 pin_ahead_pages=64', 'walk seq', 0),
    'life16':     ('pin_ahead_pages=16', 'seq stop vma vmax retire', 0),
    'life64':     ('pin_ahead_pages=64', 'seq stop vma vmax retire', 0),
    'ro64':       ('pin_ahead_pages=64', 'rowrite', 139),
    'malloc16':   ('pin_ahead_pages=16', 'ok', 0, 'malloc'),
    'malloc64':   ('pin_ahead_pages=64', 'ok', 0, 'malloc'),
}
SCRIPT = '''#!/bin/sh
cd /mnt/vm || exit 1
export CAPSTONE_VM_STATS=1 CAPSTONE_VM_STATS_MARK=1
pins() { echo 1 > /proc/sys/vm/stat_refresh; grep -E "^nr_foll_pin_(acquired|released) " /proc/vmstat | tr "\\n" " "; }
dmesg -n 1
dmesg -c > /dev/null
'''
def app(name):
    return CONFIGS[name][3] if len(CONFIGS[name]) > 3 else 'pin'


# The accounting checks compare the pins and faults of the REGION'S arena between an injected
# run and its reference, by the formula run-pin-ahead.py uses, for every module: the module's
# CAPSTONE_VM_STATS counters less the program's own pages, which the census of a module with
# the diag_admit parameter gives (dmesg, cleared after each run; the launcher's
# CAPSTONE_VM_STATS_MARK splits the arenas retired before the counters were read from those
# after) and which is zero without it. The process-wide total alone would not do: since lazy
# NONLIN grants it also holds the program's own pages, which vary with placement and with
# windows cut by the quantum. With a census the region's own census must satisfy the same
# comparison, and every run's counters are tied to the census (pages= is what the arenas
# retired after the mark still held; faults= covers every page admitted at a fault; peak= is at
# least pages=), so neither source can stand in for the other.
def script(names, instrumented):
    text = SCRIPT
    for name in names:
        params, modes = CONFIGS[name][:2]
        text += f'''insmod capstone_vm.ko {params} {'diag_admit=1' if instrumented else ''} || exit 1
for mode in {modes}; do
  echo "HOOK_PINS_BEFORE:{name}:$mode:$(pins)"
  ./capstone-vexec {app(name)}.dom "$mode" > /tmp/pin.out 2>&1
  echo "HOOK_EXIT:{name}:$mode:$?"
  echo "HOOK_PINS_AFTER:{name}:$mode:$(pins)"
  cat /tmp/pin.out
  dmesg -c | grep -E "capstone_vm: |CAPSTONE_VM_STATS_MARK" | grep -v "test unload" | sed "s/^/HOOK_DIAG:/"
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


# The module's per-arena census: one `retire` line per arena (every arena is retired by the
# process's exit at the latest) with the pages it admitted by each route and the pages swept.
RETIRE = re.compile(r'capstone_vm: retire arena 0x([0-9a-f]+)\+0x[0-9a-f]+ admit=(\d+) window=(\d+) '
                    r'scan=(\d+) touch=(\d+) eager=(\d+) released=(\d+) swept=(\d+)')
ADD_OK = re.compile(r'CAPSTONE_VM_ADD_OK slot=\d+ base=0x([0-9a-f]+) bytes=(\d+) ')


def held(c):
    return c['admit'] + c['window'] + c['scan'] + c['touch'] + c['eager'] - c['released']


def region_arena(run):
    if run.get('region') is None:
        return None
    return next((base for base, bytes_ in run['arenas'] if base <= run['region'] < base + bytes_), None)


def region_counts(run):
    """(pins, faults) of the arena holding the run's region, from its retire line: the pages
    pinned in it by every route and its demand faults; (0, 0) for a mode without a region
    (syscalls)."""
    if run.get('region') is None:
        return (0, 0)
    arena = region_arena(run)
    if arena is None or arena not in run['census']:
        return None
    c = run['census'][arena]
    return c['admit'] + c['window'] + c['scan'] + c['touch'] + c['eager'], c['admit']


def counter_counts(run):
    """(pages, faults) of the run's CAPSTONE_VM_STATS less the program's own pages from the
    census, as run-pin-ahead.py computes them: pinned, the arenas other than the region's retired
    after the launcher's mark (what pages= still counted); faults, the demand faults of every
    arena other than the region's. Zero correction without a census."""
    s = run.get('stats')
    if not s:
        return None
    arena = region_arena(run)
    pinned = sum(held(c) for base, c in run['census'].items()
                 if base != arena and base in run['after_mark'])
    faults = sum(c['admit'] for base, c in run['census'].items() if base != arena)
    return s['pages'] - pinned, s['faults'] - faults


def parse(log):
    runs, reports, ends, current = {}, {}, {}, None
    for line in log.splitlines():
        m = re.match(r'HOOK_PINS_BEFORE:(\w+):(\w+):(.*)$', line)
        if m:
            current = (m.group(1), m.group(2))
            runs[current] = {'before': pins(m.group(3)), 'stats': None, 'ok': False,
                             'region': None, 'arenas': [], 'census': {}, 'after_mark': set(),
                             'marked': False}
            continue
        if current and line.startswith('PIN_REGION:0x'):
            runs[current]['region'] = int(line[len('PIN_REGION:'):], 16); continue
        if current and (m := ADD_OK.search(line)):
            runs[current]['arenas'].append((int(m.group(1), 16), int(m.group(2)))); continue
        if current and line.startswith('HOOK_DIAG:') and 'CAPSTONE_VM_STATS_MARK' in line:
            runs[current]['marked'] = True; continue
        if current and line.startswith('HOOK_DIAG:') and (m := RETIRE.search(line)):
            base = int(m.group(1), 16)
            runs[current]['census'][base] = dict(zip(
                ('admit', 'window', 'scan', 'touch', 'eager', 'released', 'swept'),
                (int(m.group(i)) for i in range(2, 9))))
            if runs[current]['marked']:
                runs[current]['after_mark'].add(base)
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
    return runs, reports, ends


def evaluate(log, names, instrumented, completed):
    runs, reports, ends = parse(log)
    census = [json.loads(l) for l in log.splitlines() if l.startswith('{"version":1,')]
    expected_runs = sum(len(CONFIGS[n][1].split()) for n in names)
    checks = {'completed': completed, 'all_runs': len(runs) == expected_runs,
              'all_reports': sorted(reports) == sorted(names),
              'no_contract_failure': 'PIN_FAIL:' not in log and 'MALLOC_FAIL:' not in log,
              'cleanup': len(census) == len(names) and all(
                  s['live_domains'] == s['live_regions'] == s['live_bytes'] == 0 for s in census)}

    # The region's pinned pages and fault events: from the counters for every module (CSRUNV
    # entries also count quantum preemptions, which depend on timing, so steps are not used),
    # and from the census as well where there is one.
    def sources(name, mode):
        r = runs.get((name, mode))
        if not r:
            return [None]
        return [counter_counts(r)] + ([region_counts(r)] if instrumented else [])

    for (name, mode), r in runs.items():
        exit_status = CONFIGS[name][2]
        checks[f'{name}_{mode}_exit'] = r.get('exit') == exit_status and r['ok'] == (exit_status == 0)
        # The counters against the census, run by run (a killed run prints no counters).
        if instrumented and exit_status == 0:
            s = r['stats']
            checks[f'{name}_{mode}_pages_census'] = bool(
                s and r['marked'] and
                s['pages'] == sum(held(c) for base, c in r['census'].items() if base in r['after_mark']))
            checks[f'{name}_{mode}_faults_census'] = bool(
                s and r['census'] and s['faults'] >= sum(c['admit'] for c in r['census'].values()))
            checks[f'{name}_{mode}_peak'] = bool(s and s['peak'] >= s['pages'])
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
            # No fault is resolved (no GUP: it can sleep) once one is pending.
            checks[f'{name}_no_gup_while_pending'] = rep['gup_pending'] == 0
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
    # uninjected run with the same effective window. A condition raised
    # after a demand GUP stops the pin-ahead window (window 1); one raised
    # before resolution leaves the fault to the launcher's CV_RESOLVE, whose
    # window is the full one. A real quantum end or signal can also fall
    # between a fault and its resolution in CV_STEP, whatever was injected:
    # the module counts those faults (launcher_first). Each pins up to
    # window - reference window pages more than the reference would have at
    # that fault. Each such page is either one the walk never touches (one
    # more pin) or one whose own fault never happens (one fewer fault event),
    # never both: the two deltas share that budget. Without such a fault pins
    # and fault events must match exactly. Every source must hold it.
    for name, ref in (('resched2', 'base1'), ('signal2', 'base1'), ('deadline2', 'base1'),
                      ('resched3', 'base1'), ('signal3', 'base1'),
                      ('resched1', 'base1'), ('signal1', 'base1'), ('deadline1', 'base1'),
                      ('resched4', 'base16'), ('signal4', 'base16'), ('deadline4', 'base16')):
        window, ref_window = (int(re.search(r'pin_ahead_pages=(\d+)', CONFIGS[n][0] +
                                            ' pin_ahead_pages=1').group(1)) for n in (name, ref))
        late = reports.get(name, {}).get('launcher_first', 0) if window > ref_window else 0
        for mode in CONFIGS[name][1].split():
            checks[f'{name}_{mode}_accounting'] = all(
                bool(got and want and got[0] >= want[0] and got[1] <= want[1] and
                     (got[0] - want[0]) + (want[1] - got[1]) <= late * (window - ref_window))
                for got, want in zip(sources(name, mode), sources(ref, mode)))
    # Partial GUP results leave the rest to later faults: every page the walk
    # touched still ends pinned (at least window 1's pins), after more faults
    # than the full window takes.
    checks['short64_walk_pins'] = all(
        bool(got and low and full and got[0] >= low[0] and got[1] > full[1])
        for got, low, full in zip(sources('short64', 'walk'), sources('base1', 'walk'),
                                  sources('base64', 'walk')))
    return checks, runs, reports, ends


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluate', type=Path, help='a saved serial.log: run the checks on it, no guest')
    p.add_argument('--instrumented', action='store_true', help='with --evaluate: the module had diag_admit')
    p.add_argument('--qemu', type=Path)
    p.add_argument('--images', type=Path)
    p.add_argument('--adapter', type=Path, help='capstone-vexec from here')
    p.add_argument('--module', type=Path, help='the CV_TEST_HOOKS build')
    p.add_argument('--application', type=Path, help='pin.dom')
    p.add_argument('--malloc-application', type=Path, help='malloc.dom; without it the malloc configurations are skipped')
    p.add_argument('--work', type=Path)
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    if a.evaluate:
        log = a.evaluate.read_text(errors='replace').replace('\r', '')
        with_malloc = 'HOOK_PINS_BEFORE:malloc16:' in log
        names = [n for n in CONFIGS if app(n) == 'pin' or with_malloc]
        checks = evaluate(log, names, a.instrumented, True)[0]
        for name, ok in checks.items(): print(('PASS ' if ok else 'FAIL ') + name)
        return 0 if all(checks.values()) else 1
    for name in ('qemu', 'images', 'adapter', 'module', 'application', 'work'):
        if getattr(a, name) is None:
            p.error(f'--{name} is required')
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
    instrumented = b'diag_admit' in inputs['module'].read_bytes()
    gate = script(names, instrumented)
    (stage/'gate.sh').write_text(gate)
    run = subprocess.run([os.environ.get('PYTHON', 'python3'),
        str(HERE/'run-staged.py'), '--qemu', str(a.qemu), '--images', str(a.images),
        '--stage', str(stage), '--work', str(a.work/'guest'), '--disk-mib', '64',
        '--timeout', str(a.timeout)])
    log = (a.work/'guest/serial.log').read_text(errors='replace').replace('\r', '')
    checks, runs, reports, ends = evaluate(log, names, instrumented, run.returncode == 0)
    checks['inputs_unchanged'] = hashes == {
        k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    result = {'schema': 1, 'status': 'PASS' if all(checks.values()) else 'FAIL',
              'checks': checks, 'instrumented': instrumented, 'reports': reports, 'pins_at_rmmod': ends,
              'runs': {f'{n}:{m}': {k: (sorted(v) if isinstance(v, set) else v) for k, v in r.items()}
                       for (n, m), r in runs.items()}, 'sha256': hashes,
              'source_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), HERE/'pin-ahead-contract.c',
                                          HERE/'run-staged.py', HERE/'module/capstone_vm.c']},
              'gate_sha256': hashlib.sha256(gate.encode()).hexdigest()}
    (a.work/'result.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, ok in checks.items(): print(('PASS ' if ok else 'FAIL ') + name)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
