#!/usr/bin/env python3
"""The adapter's pin-ahead window: forward clip to the arena, stop at a pinned
page and at the VMA end, read-only (FOLL_FORCE) backing stays read-only, and
retirement releases every prefetched pin. Window 1 is the control. The module's adaptive window (fault_ahead_max) is turned off here: this
gate qualifies the fixed window page for page; run-step-hooks.py's 'adapt' configurations test
the adaptive one.

The adapter is built into the kernel, and pin_ahead_pages is read-only after boot, so every window
is one boot of the same stage: the kernel command line names the window and the gate script reads
it back from sysfs. The unpinned oracle needs a kernel built with CV_TEST_HOOKS=1 (test_full_window);
--hooks-images names the images directory of that kit.

A native task has no launcher loop. The launcher's counters (CAPSTONE_VM_STATS), its mark in the
kernel log and the adapter's census (--stats) do not exist there. A log without them grades what
the kernel's own records show, under names of their own, and lists every check that reads the
counters as not observable.

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
import memory_mode
import native_platform
import unpinned_checks

HERE = Path(__file__).resolve().parent
WINDOWS = (1, 16, 64)
MODES = ('base', 'seq', 'stop', 'vma', 'vmax', 'rowrite', 'retire')
# With diag_admit on the kernel command line, every arena's retirement prints its census
# (pages admitted at faults, by windows, by the resident scan, by CV_TOUCH, pinned whole at the
# mint; released before; swept), read after each run with the ring buffer cleared. A task's
# arenas are retired when it exits, before the launcher's wait returns, so each run's records
# are complete when the script reads them. The checks count the pins of the REGION'S arena
# alone -- the process-wide total also holds the program's own pages, which since lazy NONLIN
# grants are admitted at first touch and vary with placement (the allocator's address-hashed
# index in .bss) and with windows cut by the quantum. The console is quieted first (dmesg -n 1):
# lines written to the serial console inside a window cost more than the step's quantum and cut
# its second batch. The script takes the window from sysfs, so the PIN_EXIT lines also say that
# the kernel got the window the boot asked for. The kernel of a native kit is built without SMP
# and has no stat_refresh: its vmstat counters are exact.
SCRIPT = '''#!/bin/sh
cd /mnt/vm || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1
pins() { [ ! -e /proc/sys/vm/stat_refresh ] || echo 1 > /proc/sys/vm/stat_refresh; grep -E "^nr_foll_pin_(acquired|released) " /proc/vmstat | tr "\\n" " "; }
dmesg -n 1
@ADMIT@
w=$(cat /sys/module/capstone_vm/parameters/pin_ahead_pages)
@LIVENESS@dmesg -c > /dev/null
@MODE_ECHO@
for mode in ''' + ' '.join(MODES) + '''; do
  ./capstone-vexec pin.dom "$mode" > /tmp/pin.out 2>&1
  echo PIN_EXIT:$w:$mode:$?
  cat /tmp/pin.out
  dmesg -c | grep "capstone_vm: " | sed "s/^/PIN_DIAG:/"
done
echo "PIN_PINS_END:$w:$(pins)"
echo VIRTUAL_STAGED_DONE
'''
# Per mode, the region's arena: pages pinned in it (demand faults, windows and scans), the
# demand faults, and the pages swept at its retirement -- by the munmap in retire mode, at the
# process's exit otherwise, so in every mode every pin is released; see pin-ahead-contract.c.
EXPECT_REGION = {
    'base':   lambda w: (0, 0, 0),
    'seq':    lambda w: (min(w, 32), 1, min(w, 32)),
    'stop':   lambda w: (min(w, 28) + min(w, 4), 2, min(w, 28) + min(w, 4)),
    'vma':    lambda w: (min(w, 8) + min(w, 24), 2, min(w, 8) + min(w, 24)),
    'vmax':   lambda w: (min(w, 8), 1, min(w, 8)),
    'retire': lambda w: (min(w, 32), 1, min(w, 32)),
}
# The module's counters: what the mode adds over "base" in CAPSTONE_VM_STATS pages (pinned when
# the counters were read), peak and faults, beyond the program's own pages (the census of every
# arena but the region's, zero without a census). For retire the region is gone when the
# counters are read and the peak is not additive (base reaches its own peak after the munmap),
# so only the pins left and the one fault are compared; the window's size there is the one seq
# shows.
EXPECT = {
    'seq':    lambda w: (min(w, 32), min(w, 32), 1),
    'stop':   lambda w: (min(w, 28) + min(w, 4), min(w, 28) + min(w, 4), 2),
    'vma':    lambda w: (min(w, 8) + min(w, 24), min(w, 8) + min(w, 24), 2),
    'vmax':   lambda w: (min(w, 8), min(w, 8), 1),
    'retire': lambda w: (0, None, 1),
}

RETIRE = re.compile(r'capstone_vm: retire arena 0x([0-9a-f]+)\+0x[0-9a-f]+ admit=(\d+) window=(\d+) '
                    r'scan=(\d+) touch=(\d+) eager=(\d+) released=(\d+) swept=(\d+)')
# An arena's base and size. The launcher of the module route printed every grant (ADD_OK). A
# native task has no such line, and its arenas are read from the kernel's retire records: every
# arena is retired at the task's exit at the latest.
ADD_OK = re.compile(r'CAPSTONE_VM_ADD_OK slot=\d+ base=0x([0-9a-f]+) bytes=(\d+) ')
RETIRED = re.compile(r'capstone_vm: retire arena 0x([0-9a-f]+)\+0x([0-9a-f]+) ')
FIELDS = ('admit', 'window', 'scan', 'touch', 'eager', 'released', 'swept')


def held(c):
    return c['admit'] + c['window'] + c['scan'] + c['touch'] + c['eager'] - c['released']


def region_arena(run):
    if run['region'] is None:
        return None
    return next((base for base, bytes_ in run['arenas'] if base <= run['region'] < base + bytes_), None)


def region_counts(run):
    """(pins, faults, swept) of the arena holding the run's region, from its retire line."""
    arena = region_arena(run)
    c = run['census'].get(arena) if arena is not None else None
    if c is None:
        return None
    return c['admit'] + c['window'] + c['scan'] + c['touch'] + c['eager'], c['admit'], c['swept']


def own_pages(run):
    """The program's own pages in the counters: (pinned when the counters were read, demand
    faults) over every arena but the region's. Pinned: the census of the arenas retired after the
    launcher's mark, which is what pages= still counted. Zero without a census."""
    arena = region_arena(run)
    pinned = sum(held(c) for base, c in run['census'].items()
                 if base != arena and base in run['after_mark'])
    faults = sum(c['admit'] for base, c in run['census'].items() if base != arena)
    return pinned, faults


def parse(log):
    """Each PIN_EXIT line is followed by that run's output (its stats and region included) and
    then the module's diag lines of that run, the launcher's mark among them."""
    runs, current = {}, None
    for line in log.splitlines():
        m = re.match(r'PIN_EXIT:(\d+):(\w+):(\d+)$', line)
        if m:
            current = (int(m.group(1)), m.group(2))
            runs[current] = {'exit': int(m.group(3)), 'stats': None, 'ok': False, 'region': None,
                             'arenas': [], 'census': {}, 'after_mark': set(), 'marked': False}
        elif current and line.startswith('CAPSTONE_VM_STATS '):
            runs[current]['stats'] = dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
        elif current and line == 'PIN_OK:' + current[1]:
            runs[current]['ok'] = True
        elif current and line.startswith('PIN_REGION:0x'):
            runs[current]['region'] = int(line[len('PIN_REGION:'):], 16)
        elif current and (m := ADD_OK.search(line)):
            runs[current]['arenas'].append((int(m.group(1), 16), int(m.group(2))))
        elif current and line.startswith('PIN_DIAG:'):
            if 'CAPSTONE_VM_STATS_MARK' in line:
                runs[current]['marked'] = True
            elif (m := RETIRE.search(line)):
                base = int(m.group(1), 16)
                runs[current]['census'][base] = dict(zip(FIELDS, (int(m.group(i)) for i in range(2, 9))))
                if runs[current]['marked']:
                    runs[current]['after_mark'].add(base)
                arena = (base, int(RETIRED.search(line).group(2), 16))
                if arena not in runs[current]['arenas']:
                    runs[current]['arenas'].append(arena)
    return runs


# An unpinned run's independent pin witness is the kernel's FOLL_PIN counter, which must not move
# during the whole run. It is worth that zero only if the counters count, so every boot first
# runs the pthread contract: on the module route a thread's supervisor frame was pinned even in
# unpinned mode, so that launch acquired and released pins. The check keeps that requirement. A
# boot whose liveness launch fails, or acquires no pin, has no control for its zero, and the
# lines PIN_LIVE_OUT show what the launch printed last.
LIVENESS = '''./capstone-vexec pthread.dom > /tmp/live.out 2>&1
echo "PIN_LIVE_STATUS:$w:$?"
echo "PIN_LIVE_MODE:$w:$(cat /sys/module/capstone_vm/parameters/unpinned 2>/dev/null || echo A)"
echo "PIN_PINS_LIVE:$w:$(pins)"
tail -n 4 /tmp/live.out | sed "s/^/PIN_LIVE_OUT:$w:/"
'''


def foll_pins(text):
    v = dict(re.findall(r'nr_foll_pin_(acquired|released) (\d+)', text))
    return (int(v['acquired']), int(v['released'])) if len(v) == 2 else None


def kernel_accounted(run):
    """unpinned_checks.accounted() without the launcher's counters, from the kernel's records alone.

    The kernel numbers a context's fault events from 1 without a gap. Each event lies in a retired
    arena with a load or store page fault cause, every arena's retire record agrees with its own
    events and windows, no page is resolved twice, and at least one fault was resolved. That the
    last number equals the launcher's public total is the part this cannot say.
    """
    events = run.get('raw_events', [])
    windows = run.get('raw_windows', [])
    retired = run.get('raw_retire', {})
    return bool(retired and events and
        [e['number'] for e in events] == list(range(1, len(events) + 1)) and
        all(e['arena'] in retired and e['cause'] in (13, 15) and
            any(base == e['arena'] and base <= e['address'] < base + size
                for base, size in run['arenas']) for e in events) and
        all(v['events'] == sum(e['arena'] == base for e in events) and
            v['ahead'] == sum(w[2] for w in windows if w[0] == base)
            for base, v in retired.items()) and
        sum(v['ahead'] for v in retired.values()) == sum(w[2] for w in windows) and
        unpinned_checks.disjoint(run) and
        sum(v['resolves'] for v in retired.values()) > 0)


def evaluate(log, completed, instrumented, unpinned=False):
    runs = parse(log)
    census = [json.loads(l) for l in log.splitlines() if l.startswith('{"version":1,')]
    checks = {'completed': completed,
              'all_runs': len(runs) == len(WINDOWS) * len(MODES),
              'memory_mode_verified': memory_mode.verified(log, unpinned, len(WINDOWS)),
              'no_contract_failure': 'PIN_FAIL:' not in log}
    # A check that reads the launcher's counters, its mark or the adapter's census is graded when
    # the log has the counters. A native task prints none: the check's name then goes to
    # not_observable, and what the kernel's own records show is graded under a name of its own.
    counters = 'CAPSTONE_VM_STATS ' in log
    not_observable = []

    def counted(name, ok):
        if counters:
            checks[name] = ok
        else:
            not_observable.append(name)

    def replaced(name):
        checks.pop(name, None)
        if name in not_observable:
            not_observable.remove(name)

    region, deltas = {}, {}
    for w in WINDOWS:
        base = runs.get((w, 'base'))
        base_ok = bool(base and base['exit'] == 0 and base['ok'] and base['stats'])
        counted(f'w{w}_base', base_ok)
        # The region's own census against the window's arithmetic (instrumented module).
        if instrumented:
            for mode, expect in EXPECT_REGION.items():
                r = runs.get((w, mode))
                got = region_counts(r) if r else None
                ok = bool(r and r['exit'] == 0 and r['ok'] and got is not None)
                if ok:
                    region[f'w{w}_{mode}'] = got
                    ok = got == expect(w)
                checks[f'w{w}_{mode}_region'] = ok
        # The module's counters: the same assertion for every module. What the mode adds over
        # base in pages, peak and faults is the window's arithmetic plus the difference in the
        # program's own pages, which the census gives and which is zero without one.
        for mode, expect in EXPECT.items():
            r = runs.get((w, mode))
            ok = bool(r and r['exit'] == 0 and r['ok'] and r['stats'] and base_ok)
            if ok:
                s, b = r['stats'], base['stats']
                own_r, own_b = own_pages(r), own_pages(base)
                got = (s['pages'] - b['pages'], s['peak'] - b['peak'], s['faults'] - b['faults'])
                want = expect(w)
                want = (want[0] + own_r[0] - own_b[0],
                        None if want[1] is None else want[1] + own_r[0] - own_b[0],
                        want[2] + own_r[1] - own_b[1])
                deltas[f'w{w}_{mode}'] = got
                ok = all(e is None or g == e for g, e in zip(got, want))
            counted(f'w{w}_{mode}', ok)
        # Instrumented: pages= of every completed run is exactly what the arenas retired after
        # the launcher's mark still held (the counter against the whole census).
        if instrumented:
            for mode in EXPECT_REGION:
                r = runs.get((w, mode))
                counted(f'w{w}_{mode}_pages_census', bool(
                    r and r['stats'] and r['marked'] and
                    r['stats']['pages'] == sum(held(c) for base, c in r['census'].items()
                                               if base in r['after_mark'])))
        r = runs.get((w, 'rowrite'))
        checks[f'w{w}_rowrite_faults'] = bool(r and r['exit'] == 139 and not r['ok'])
    checks['rowrite_reached_store'] = log.count('PIN_ROWRITE_ATTEMPT') == len(WINDOWS)
    checks['denials_are_store_faults'] = re.findall(
        r'^capstone-exec: domain fault cause=(\d+) ', log, re.M) == ['15'] * len(WINDOWS)
    counted('cleanup', len(census) == len(WINDOWS) and all(
        s['live_domains'] == s['live_regions'] == s['live_bytes'] == 0 for s in census))
    if unpinned:
        unpinned_checks.attach(log, runs)
        # Every window is its own boot, and each boot proves its own counters live.
        live = {int(w): foll_pins(t) for w, t in re.findall(r'^PIN_PINS_LIVE:(\d+):(.*)$', log, re.M)}
        ends = {int(w): foll_pins(t) for w, t in re.findall(r'^PIN_PINS_END:(\d+):(.*)$', log, re.M)}
        checks['foll_pin_counters_live'] = sorted(live) == sorted(WINDOWS) and all(
            bool(live[w] and live[w][0] == live[w][1] > 0 and
                 f'PIN_LIVE_STATUS:{w}:0\n' in log and f'PIN_LIVE_MODE:{w}:Y\n' in log) for w in WINDOWS)
        checks['no_payload_pins'] = sorted(live) == sorted(ends) == sorted(WINDOWS) and all(
            live[w] is not None and ends[w] == live[w] for w in WINDOWS)
        for w in WINDOWS:
            for mode, expect in EXPECT_REGION.items():
                r = runs.get((w, mode))
                counts = unpinned_checks.region_counts(r) if r else None
                checks.pop(f'w{w}_{mode}_region', None)
                checks[f'w{w}_{mode}_window_event_disjoint'] = bool(r and unpinned_checks.disjoint(r) and r.get('raw_events'))
                counted(f'w{w}_{mode}_resolved_region', bool(r and r['exit'] == 0 and r['ok'] and
                    counts == expect(w)[:2] and unpinned_checks.accounted(r)))
                if not counters:
                    checks[f'w{w}_{mode}_resolved_region_kernel'] = bool(r and r['exit'] == 0 and r['ok'] and
                        counts == expect(w)[:2] and kernel_accounted(r))
                if mode in ('vma', 'vmax'):
                    # pages 8.. are another VMA: no window planned from the first one may reach it
                    checks[f'w{w}_{mode}_window_in_vma'] = bool(r and r['region'] is not None and
                        unpinned_checks.windows_stay_below(r, r['region'] + 8 * 4096))
                if mode in EXPECT:
                    replaced(f'w{w}_{mode}')
                    counted(f'w{w}_{mode}_unpinned_accounting', bool(r and
                        unpinned_checks.counter_counts(r) == expect(w)[:2] and unpinned_checks.accounted(r)))
    return checks, deltas, region, runs, not_observable


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluate', type=Path, help='a saved serial.log: run the checks on it, no guest')
    p.add_argument('--instrumented', action='store_true', help='with --evaluate: the kernel had diag_admit')
    p.add_argument('--unpinned', action='store_true', help='require unpinned mode and its resolution oracle (the native adapter has no other mode)')
    p.add_argument('--qemu', type=Path)
    p.add_argument('--images', type=Path, help="the kit's images directory; booted unless --hooks-images is given")
    p.add_argument('--adapter', type=Path)
    p.add_argument('--hooks-images', type=Path,
                   help='images directory of a kit whose kernel Image was built with CV_TEST_HOOKS=1 (with its '
                        'fw_jump.elf and rootfs.ext2); every boot of this gate then uses it instead of --images. '
                        '--unpinned needs it for test_full_window')
    p.add_argument('--application', type=Path)
    p.add_argument('--pthread-application', type=Path, help='pthread.dom; the liveness launch of the FOLL_PIN counters')
    p.add_argument('--work', type=Path)
    p.add_argument('--timeout', type=int, default=600, help='seconds for each boot')
    a = p.parse_args()
    if a.evaluate:
        log = a.evaluate.read_text(errors='replace').replace('\r', '')
        checks, deltas, region, _, not_observable = evaluate(log, True, a.instrumented, a.unpinned)
        report(checks, deltas, region, not_observable)
        return 0 if all(checks.values()) else 1
    for name in ('qemu', 'images', 'adapter', 'application', 'work'):
        if getattr(a, name) is None:
            p.error(f'--{name} is required')
    if a.unpinned and not a.pthread_application:
        p.error('--unpinned requires --pthread-application')
    images = a.hooks_images or a.images
    kernel = (images/'Image').read_bytes()
    instrumented = b'diag_admit' in kernel
    if a.unpinned and b'test_full_window' not in kernel:
        p.error('--unpinned requires --hooks-images: a kernel built with CV_TEST_HOOKS=1, for deterministic window policy')
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'; stage.mkdir()
    inputs = {'qemu': a.qemu, 'application': a.application,
              'launcher': a.adapter/native_platform.LAUNCHER,
              **{n: images/n for n in ('Image', 'fw_jump.elf', 'rootfs.ext2')}}
    if a.unpinned:
        inputs['pthread'] = a.pthread_application
        shutil.copy2(a.pthread_application, stage/'pthread.dom')
    hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    for src, name in ((inputs['application'], 'pin.dom'),
                      (inputs['launcher'], native_platform.LAUNCHER)):
        shutil.copy2(src, stage/name)
    gate = (SCRIPT.replace('@ADMIT@', native_platform.admit())
                  .replace('@LIVENESS@', LIVENESS if a.unpinned else '')
                  .replace('@MODE_ECHO@', memory_mode.echo('PIN_')))
    (stage/'gate.sh').write_text(gate)
    log, completed, kernel_arguments = '', True, {}
    for w in WINDOWS:
        kernel_arguments[w] = window_arguments(w, instrumented, a.unpinned)
        run = subprocess.run([os.environ.get('PYTHON', 'python3'),
            str(HERE/'run-staged.py'), '--qemu', str(a.qemu), '--images', str(images),
            '--stage', str(stage), '--work', str(a.work/f'guest-w{w}'), '--disk-mib', '64',
            '--timeout', str(a.timeout),
            *(word for argument in kernel_arguments[w] for word in ('--kernel-arg', argument))])
        completed = completed and run.returncode == 0
        log += (a.work/f'guest-w{w}/serial.log').read_text(errors='replace').replace('\r', '')
    # The three transcripts in window order, as one file --evaluate can read.
    (a.work/'serial.log').write_text(log)
    checks, deltas, region, runs, not_observable = evaluate(log, completed, instrumented, a.unpinned)
    checks['inputs_unchanged'] = hashes == {
        k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    result = {'schema': 1, 'status': 'PASS' if all(checks.values()) else 'FAIL',
              'checks': checks, 'not_observable': not_observable,
              'instrumented': instrumented, 'unpinned': a.unpinned,
              'kernel_arguments': {str(w): words for w, words in kernel_arguments.items()},
              'resolved_pages_and_events': {f'{w}:{m}': unpinned_checks.region_counts(r)
                                           for (w, m), r in runs.items()} if a.unpinned else None,
              'deltas_pages_peak_faults': deltas, 'region_pins_faults_swept': region,
              'runs': {f'{w}:{m}': {k: (sorted(v) if isinstance(v, set) else v) for k, v in r.items()}
                       for (w, m), r in runs.items()},
              'sha256': hashes,
              'source_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), HERE/'pin-ahead-contract.c',
                                          HERE/'run-staged.py', HERE/'unpinned_checks.py']},
              'gate_sha256': hashlib.sha256(gate.encode()).hexdigest(),
              'scope': 'one hart, private anonymous arenas, windows ' + ', '.join(map(str, WINDOWS))}
    (a.work/'result.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    report(checks, deltas, region, not_observable)
    return 0 if all(checks.values()) else 1


def window_arguments(window, instrumented, unpinned):
    """The kernel words of one window's boot, where the module route passed insmod parameters."""
    words = [f'capstone_vm.pin_ahead_pages={window}', 'capstone_vm.fault_ahead_max=0']
    if instrumented:
        words.append('capstone_vm.diag_admit=1')
    if unpinned:
        words.append('capstone_vm.test_full_window=1')
    return words


def report(checks, deltas, region, not_observable):
    for name, ok in checks.items(): print(('PASS ' if ok else 'FAIL ') + name)
    for name in not_observable: print('NOT OBSERVABLE ' + name)
    print('deltas (pages, peak, faults):', json.dumps(deltas))
    print('region (pins, faults, swept):', json.dumps(region))


if __name__ == '__main__':
    raise SystemExit(main())
