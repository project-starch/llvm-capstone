#!/usr/bin/env python3
"""CV_STEP's stop checks and the pin-ahead window, under injected conditions.

Needs a kit whose kernel Image was built with the adapter's test hooks (make
CV_TEST_HOOKS=1 Image), passed with --hooks-images; the production kernel has none of the
hooks. A hook injects need_resched, a signal (TIF_NOTIFY_SIGNAL), an expired
deadline or SIGKILL at one point of CV_STEP: right after a demand GUP (1), in
the interrupt-enabled window before a later CSRUNV (2), before the first
CSRUNV of a step (3), or after CSRUNV returns a fault, before it is
resolved (4). The adapter counts injections, the stops they caused, any CSRUNV
entered and any GUP requested while one was pending (violations), plus every
GUP request and its size, every stop check made with interrupts on, and every
CSRUNV entry. It also counts the stops nobody injected (a real
quantum end, signal or reschedule) where they change what a fault pins, and tags every
fault-ahead window with the route that resolved its fault (by=step or by=resolve). /proc/vmstat's FOLL_PIN counters show
that every pin taken was released, including over the pin-ahead modes,
retirement, a dying domain and, with --malloc-application (malloc.dom from
malloc-contract.c), mremap growth, shrinking and failure.

The adapter is built into the kernel and its hook parameters are read-only after boot, so every
configuration is one boot: its parameters stand on the kernel command line where the module
route passed them to insmod.

Two things the module route read do not exist for a native task. The adapter printed its hook
counters when the module was unloaded (HOOK_REPORT), and a built-in adapter is never unloaded.
The launcher printed its counters (CAPSTONE_VM_STATS), a mark in the kernel log, every grant and
the adapter's census (--stats), and a native task has no launcher loop. A log without them
lists every check that reads them as not observable. Where such a check also reads the
kernel's own records, that part is graded under the check's name with _kernel appended.

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
# name: (adapter parameters, modes, expected exit status[, application])
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
    # The adaptive fault-ahead window (capstone_vm.c fault_ahead_max), the module's default: every
    # other configuration turns it off (fault_ahead_max=0, added in parameters()), since each
    # qualifies a FIXED window page for page. 'adapt' walks with it (the 32-page region, and
    # pin.dom's 512-page 'stream' region); 'adaptres' raises a reschedule after every demand GUP,
    # so each window must stop before its first batch.
    # Policy-only counts suppress scheduler/budget cuts inside the window, not page or VMA
    # checks. adaptdirect and every stop-injection case keep the real stop checks enabled.
    'adapt':      ('test_full_window=1', 'walk stream', 0),
    'adaptdirect': ('', 'walk stream', 0),
    'adaptres':   ('test_inject=1 test_inject_at=1', 'walk', 0),
    'adaptshort': ('test_full_window=1 test_short_gup=1', 'walk', 0),
    'adaptclip':  ('test_full_window=1', 'clip clipx', 0),
    'adaptro':    ('test_full_window=1', 'clipwrite', 139),
    'adaptnonlin': ('test_full_window=1', 'nonlin', 0),
    'adaptgaps':  ('test_full_window=1', 'gaps', 0),
    'adaptinterleave': ('test_full_window=1', 'interleave', 0),
}
SCRIPT = '''#!/bin/sh
cd /mnt/vm || exit 1
pins() { [ ! -e /proc/sys/vm/stat_refresh ] || echo 1 > /proc/sys/vm/stat_refresh; grep -E "^nr_foll_pin_(acquired|released) " /proc/vmstat | tr "\\n" " "; }
ahead() { for p in fault_ahead_windows fault_ahead_pages; do printf "$p="; cat /sys/module/capstone_vm/parameters/$p 2>/dev/null || echo unavailable; done; }
dmesg -n 1
'''
def app(name):
    return CONFIGS[name][3] if len(CONFIGS[name]) > 3 else 'pin'


# The accounting checks compare the pins and faults of the REGION'S arena between an injected
# run and its reference. The kernel's records give them: with diag_admit, every arena's
# retirement prints its census and every fault event and window prints a line (dmesg, cleared
# after each run; a task's arenas are retired when it exits, before the launcher's wait
# returns). The process-wide total alone would not do: since lazy NONLIN grants it also holds
# the program's own pages, which vary with placement and with windows cut by the quantum. The
# launcher of the module route printed a second source, its counters less the program's own
# pages; a log that has them must satisfy every comparison from both sources.
# The kernel's FOLL_PIN counters (nr_foll_pin_acquired / _released) are the independent witness that
# an unpinned run takes no payload pin: they must read the same after a configuration's last run
# as before its first. The witness is worth that only if the counters count, so one more boot
# of the same kernel, with no hook parameter, runs the pthread contract: on the module route a
# thread's supervisor frame was pinned even in unpinned mode (CSRUNV names it physically), so
# that launch acquired and released pins. A configuration's own boot cannot run it, because
# its injected conditions would reach that launch too. Its lines are HOOK_LIVE_*, not a
# *MEMORY_MODE line.
LIVE = '@live'
LIVENESS = '''./capstone-vexec pthread.dom > /tmp/live.out 2>&1
echo "HOOK_LIVE_STATUS:$?"
echo "HOOK_LIVE_MODE:$(cat /sys/module/capstone_vm/parameters/unpinned 2>/dev/null || echo A)"
echo "HOOK_PINS_LIVE:$(pins)"
tail -n 4 /tmp/live.out | sed "s/^/HOOK_LIVE_OUT:/"
'''


def script(name):
    """The gate script of one boot: a configuration's runs, or the liveness launch."""
    text = SCRIPT + native_platform.admit() + '\n'
    if name == LIVE:
        return text + LIVENESS + 'echo VIRTUAL_STAGED_DONE\n'
    modes = CONFIGS[name][1]
    return text + f'''dmesg -c > /dev/null
{memory_mode.echo('HOOK_')}
echo "HOOK_UNPINNED:{name}:$(cat /sys/module/capstone_vm/parameters/unpinned 2>/dev/null || echo unavailable)"
for mode in {modes}; do
  echo "HOOK_PINS_BEFORE:{name}:$mode:$(pins)"
  echo "HOOK_AHEAD_BEFORE:{name}:$mode:$(ahead | tr '\\n' ' ')"
  ./capstone-vexec {app(name)}.dom "$mode" > /tmp/pin.out 2>&1
  echo "HOOK_EXIT:{name}:$mode:$?"
  echo "HOOK_AHEAD_AFTER:{name}:$mode:$(ahead | tr '\\n' ' ')"
  echo "HOOK_PINS_AFTER:{name}:$mode:$(pins)"
  cat /tmp/pin.out
  dmesg -c | grep "capstone_vm: " | sed "s/^/HOOK_DIAG:/"
done
echo "HOOK_PINS_END:{name}:$(pins)"
echo VIRTUAL_STAGED_DONE
'''


def parameters(name, instrumented, extra=''):
    """The kernel words of one boot, where the module route passed insmod parameters."""
    if name == LIVE:
        return []
    words = CONFIGS[name][0].split()
    if not name.startswith('adapt'):
        words.append('fault_ahead_max=0')
    if instrumented:
        words.append('diag_admit=1')
    return ['capstone_vm.' + word for word in words + extra.split()]


def pins(text):
    v = dict(re.findall(r'nr_foll_pin_(acquired|released) (\d+)', text))
    return int(v['acquired']), int(v['released'])


# The module's per-arena census: one `retire` line per arena (every arena is retired by the
# process's exit at the latest) with the pages it admitted by each route and the pages swept.
RETIRE = re.compile(r'capstone_vm: retire arena 0x([0-9a-f]+)\+0x[0-9a-f]+ admit=(\d+) window=(\d+) '
                    r'scan=(\d+) touch=(\d+) eager=(\d+) released=(\d+) swept=(\d+)(?: faulted=(\d+))?')
# An arena's base and size. The launcher of the module route printed every grant (ADD_OK). A
# native task has no such line, and its arenas are read from the kernel's retire records: every
# arena is retired at the task's exit at the latest.
ADD_OK = re.compile(r'CAPSTONE_VM_ADD_OK slot=\d+ base=0x([0-9a-f]+) bytes=(\d+) ')
RETIRED = re.compile(r'capstone_vm: retire arena 0x([0-9a-f]+)\+0x([0-9a-f]+) ')


def held(c):
    return c['admit'] + c['window'] + c['scan'] + c['touch'] + c['eager'] - c['released']


def fault_events(run, c):
    """The demand fault events one arena's retire line accounts for. A pinning module admits a page
    at every fault, so admit= is the count. The unpinned mode (HOOK_UNPINNED: Y) pins and admits
    nothing: its count is the line's faulted= (the events CV_STEP received on the arena, counted
    apart from the process total they are subtracted from), and a module that prints none gives None, so
    the comparisons that need it fail instead of falling back to the process total, which holds the
    program's own pages (a lookup of the allocator's hashed .bss index that lands on the page of
    the launch record faults once less than one that does not: -1 step and -1 fault, by placement)."""
    return c['faulted'] if run.get('unpinned') else c['admit']


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
    faults = fault_events(run, c)
    return None if faults is None else (c['admit'] + c['window'] + c['scan'] + c['touch'] + c['eager'], faults)


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
    events = [fault_events(run, c) for base, c in run['census'].items() if base != arena]
    if None in events:
        return None
    return s['pages'] - pinned, s['faults'] - sum(events)


def parse(log):
    runs, reports, ends, current = {}, {}, {}, None
    live = re.findall(r'^HOOK_PINS_LIVE:(.*)$', log, re.M)
    try:
        ends[LIVE] = pins(live[0]) if len(live) == 1 else None
    except KeyError:
        ends[LIVE] = None
    unpinned = {}
    for line in log.splitlines():
        m = re.match(r'HOOK_UNPINNED:(\w+):(\w+)$', line)
        if m:
            unpinned[m.group(1)] = m.group(2) == 'Y'
            continue
        m = re.match(r'HOOK_PINS_BEFORE:(\w+):(\w+):(.*)$', line)
        if m:
            current = (m.group(1), m.group(2))
            runs[current] = {'before': pins(m.group(3)), 'stats': None, 'ok': False,
                             'unpinned': unpinned.get(m.group(1), False),
                             'region': None, 'arenas': [], 'census': {}, 'after_mark': set(),
                             'marked': False, 'ahead': [], 'linear': {}}
            continue
        m = re.match(r'HOOK_AHEAD_(BEFORE|AFTER):(\w+):(\w+):(.*)$', line)
        if m and current == (m.group(2), m.group(3)):
            runs[current]['ahead_' + m.group(1).lower()] = dict(
                (k, int(v)) for k, v in re.findall(r'(fault_ahead_\w+)=(\d+)', m.group(4)))
            continue
        if current and line.startswith('HOOK_DIAG:') and (m := re.search(
                r'capstone_vm: fault-ahead arena 0x([0-9a-f]+) page (\d+) want (\d+) pinned (\d+) fault 0x([0-9a-f]+) by=(\w+)', line)):
            # (arena, page, want, pinned, fault address, route): the route is 'step' when CV_STEP's
            # own loop resolved the fault (where an injected condition stops the window) and
            # 'resolve' when the launcher's CV_RESOLVE did.
            runs[current]['ahead'].append((int(m.group(1), 16), *(int(m.group(i)) for i in (2, 3, 4)),
                                           int(m.group(5), 16), m.group(6)))
            continue
        if current and line.startswith('PIN_REGION:0x'):
            runs[current]['region'] = int(line[len('PIN_REGION:'):], 16); continue
        if current and line.startswith('PIN_SPAN:'):
            runs[current]['span'] = int(line[len('PIN_SPAN:'):]) * 4096; continue
        if current and line == 'PIN_ROWRITE_ATTEMPT':
            runs[current]['rowrite'] = True; continue
        if current and (m := ADD_OK.search(line)):
            base = int(m.group(1), 16)
            runs[current]['arenas'].append((base, int(m.group(2))))
            linear = re.search(r'\blinear=(\d+)', line)
            if linear:
                runs[current]['linear'][base] = int(linear.group(1))
            continue
        if current and line.startswith('HOOK_DIAG:') and 'CAPSTONE_VM_STATS_MARK' in line:
            runs[current]['marked'] = True; continue
        if current and line.startswith('HOOK_DIAG:') and (m := RETIRE.search(line)):
            base = int(m.group(1), 16)
            runs[current]['census'][base] = dict(zip(
                ('admit', 'window', 'scan', 'touch', 'eager', 'released', 'swept'),
                (int(m.group(i)) for i in range(2, 9))))
            runs[current]['census'][base]['faulted'] = int(m.group(9)) if m.group(9) else None
            if runs[current]['marked']:
                runs[current]['after_mark'].add(base)
            arena = (base, int(RETIRED.search(line).group(2), 16))
            if arena not in runs[current]['arenas']:
                runs[current]['arenas'].append(arena)
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


def evaluate(log, names, instrumented, completed, unpinned=False):
    runs, reports, ends = parse(log)
    if unpinned:
        unpinned_checks.attach(log, runs, hooks=True)
    census = [json.loads(l) for l in log.splitlines() if l.startswith('{"version":1,')]
    expected_runs = sum(len(CONFIGS[n][1].split()) for n in names)
    # What a native task's log does not have. counters: the launcher's CAPSTONE_VM_STATS, its
    # mark, its grant lines and the adapter's census. reported: the adapter's hook counters.
    counters = 'CAPSTONE_VM_STATS ' in log
    reported = bool(reports)
    not_observable = []
    checks = {'completed': completed, 'all_runs': len(runs) == expected_runs,
              'memory_mode_verified': memory_mode.verified(log, unpinned, len(names)),
              'no_contract_failure': 'PIN_FAIL:' not in log and 'MALLOC_FAIL:' not in log}

    def graded(name, ok, observable):
        """A check that reads the launcher's counters or the hook report: graded when the log has them."""
        if observable:
            checks[name] = ok
        else:
            not_observable.append(name)

    def partly(name, ok, observable=None):
        """A check that reads them next to the kernel's own records. Without them the caller
        computes `ok` from the records alone, and it is graded as <name>_kernel."""
        if counters if observable is None else observable:
            checks[name] = ok
        else:
            not_observable.append(name)
            checks[name + '_kernel'] = ok

    accounted = unpinned_checks.accounted if counters else kernel_accounted
    graded('all_reports', sorted(reports) == sorted(names), reported)
    graded('cleanup', len(census) == len(names) and all(
        s['live_domains'] == s['live_regions'] == s['live_bytes'] == 0 for s in census), counters)

    # The region's pinned pages and fault events: from the kernel's records (the census of a
    # pinning adapter, the fault events and windows of an unpinned one), and from the launcher's
    # counters as well where the log has them (CSRUNV entries also count quantum preemptions,
    # which depend on timing, so steps are not used).
    def sources(name, mode):
        r = runs.get((name, mode))
        if not r:
            return [None]
        if unpinned:
            launcher, kernel = unpinned_checks.counter_counts(r), [unpinned_checks.region_counts(r)]
        else:
            launcher, kernel = counter_counts(r), [region_counts(r)] if instrumented else []
        return [launcher] + kernel if counters else kernel or [None]

    for (name, mode), r in runs.items():
        exit_status = CONFIGS[name][2]
        checks[f'{name}_{mode}_exit'] = r.get('exit') == exit_status and r['ok'] == (exit_status == 0)
        if exit_status != 0:
            continue    # a killed run prints no counters
        s = r['stats']
        # The counters against the census, run by run.
        if instrumented:
            graded(f'{name}_{mode}_pages_census', bool(
                s and r['marked'] and
                s['pages'] == sum(held(c) for base, c in r['census'].items() if base in r['after_mark'])),
                counters)
            graded(f'{name}_{mode}_peak', bool(s and s['peak'] >= s['pages']), counters)
        # faulted= is counted where CV_STEP receives a fault event and the process total
        # where it is resolved or returned, two sites that must agree: in the unpinned
        # mode the sum equals the total (a miss at either site shows), with a pinning module
        # the census covers every page admitted at a fault and the total may exceed it.
        if unpinned:
            partly(f'{name}_{mode}_faults_census', accounted(r))
        elif instrumented:
            events = [fault_events(r, c) for c in r['census'].values()]
            graded(f'{name}_{mode}_faults_census', bool(
                s and r['census'] and None not in events and s['faults'] >= sum(events)), counters)
    # Releases can trail a process's exit, so a pinning adapter's balance is taken at the
    # configuration's end, when no context is left: every pin Linux recorded as acquired has
    # been released.
    if not unpinned:
        for name in names:
            e = ends.get(name)
            checks[f'{name}_pins_balanced'] = bool(e and e[0] == e[1] and e[0] > 0)
    # The adapter's hook counters. With no report in the log every one of these is not
    # observable; with reports, a configuration whose report is missing fails all_reports.
    for name in names:
        rep = reports.get(name)
        if reported and rep is None:
            continue
        params = CONFIGS[name][0]

        def hook(check, value, observable=reported):
            graded(f'{name}_{check}', bool(rep is not None and value()), observable)

        if 'test_inject=' in params:
            hook('injected', lambda: rep['injections'] > 0)
            # A need_resched consumed by a reschedule inside GUP needs no stop.
            hook('every_injection_stopped', lambda: (
                rep['stops'] + rep['consumed'] == rep['injections'] and
                (rep['consumed'] == 0 or 'test_inject=1 ' in params + ' ')))
            hook('no_entry_while_pending', lambda: rep['violations'] == 0)
            # No fault is resolved (no GUP: it can sleep) once one is pending.
            hook('no_gup_while_pending', lambda: rep['gup_pending'] == 0)
        if 'test_inject_at=3' in params:
            hook('initial_stops', lambda: rep['initial_stops'] == rep['injections'])
        if not unpinned:
            hook('gup_batch_bound', lambda: 0 < rep['gup_max'] <= 16)
        # Every stop check ran with interrupts off, right before CSRUNV.
        hook('checks_irqs_off', lambda: rep['checks'] > 0 and rep['checks_irqs_on'] == 0)
        # The launcher's CSRUNV-entry count is the module's own (a stopped
        # first entry is no entry); a killed process prints no stats.
        if CONFIGS[name][2] == 0:
            hook('steps_are_entries', lambda: sum(
                r['stats']['steps'] for (n, m), r in runs.items()
                if n == name and r['stats']) == rep['entries'], reported and counters)
    if unpinned:
        graded('base64_full_fault_batches', reports.get('base64', {}).get('fault_max') == 16, reported)
        graded('short64_deferred_and_partial_faults', all(
            reports.get('short64', {}).get(k, 0) > 0 for k in ('fault_deferred', 'fault_short')), reported)
    else:
        graded('base64_full_batches', reports.get('base64', {}).get('gup_max') == 16, reported)
        graded('short64_partial_results', reports.get('short64', {}).get('short_results', 0) > 0, reported)
    # A stop changes no accounting: same pins and fault events as the
    # uninjected run with the same effective window. A condition raised
    # after a demand GUP stops the pin-ahead window (window 1); one raised
    # before resolution leaves the fault to the launcher's CV_RESOLVE, whose
    # window is the full one. A real quantum end, signal or reschedule that
    # nobody injected can also land in two places, whatever was injected, and the
    # module counts both (only stops with no injection pending):
    #  - between a fault and its resolution in CV_STEP (launcher_first): the launcher's
    #    CV_RESOLVE then pins the full window. Each such fault pins up to window -
    #    reference window pages more than the reference would have. Each such page is
    #    either one the walk never touches (one more pin) or one whose own fault never
    #    happens (one fewer fault event), never both: the two deltas share that budget.
    #  - before a batch of a pin-ahead window (real_window): the window is cut, and its
    #    pages are pinned by a later fault or fault themselves: fewer pins or more fault
    #    events, at most window - 1 pages per cut. The reference run is exposed to the
    #    same cut, which shows as the opposite delta.
    # Each delta must be explained by counted events and by nothing else: without them
    # pins and fault events must match exactly. Every source must hold it. The allowances are
    # the report's counts, so without the report the comparison has no form that reads the
    # kernel's records alone. An unpinned log calls it <name>_<mode>_reference: the pinned
    # oracle's reference comparison on the unpinned sources, where pages are the distinct
    # virtual pages resolved.
    for name, ref in (('resched2', 'base1'), ('signal2', 'base1'), ('deadline2', 'base1'),
                      ('resched3', 'base1'), ('signal3', 'base1'),
                      ('resched1', 'base1'), ('signal1', 'base1'), ('deadline1', 'base1'),
                      ('resched4', 'base16'), ('signal4', 'base16'), ('deadline4', 'base16')):
        window, ref_window = (int(re.search(r'pin_ahead_pages=(\d+)', CONFIGS[n][0] +
                                            ' pin_ahead_pages=1').group(1)) for n in (name, ref))
        late = reports.get(name, {}).get('launcher_first', 0) if window > ref_window else 0
        cut, ref_cut = (reports.get(n, {}).get('real_window', 0) for n in (name, ref))
        more = late * (window - ref_window) + ref_cut * (ref_window - 1)   # pins up, fault events down
        less = cut * (window - 1)                                          # pins down, fault events up
        for mode in CONFIGS[name][1].split():
            graded(f'{name}_{mode}_' + ('reference' if unpinned else 'accounting'), all(
                bool(got and want and
                     max(got[0] - want[0], 0) + max(want[1] - got[1], 0) <= more and
                     max(want[0] - got[0], 0) + max(got[1] - want[1], 0) <= less)
                for got, want in zip(sources(name, mode), sources(ref, mode))),
                counters and reported)
    # Exact default-policy census: windows 4, 8, 16, 32, then 64, clipped at the arena's end.
    # The sysfs counters include startup arenas. Subtract their diagnostic events, but also
    # require the WHOLE counter delta to equal the whole event census, so a missing event or
    # a broken counter cannot make a window pass. Empty/missing counters fail (including dev).
    def region_events(r):
        arena = region_arena(r)
        return [e for e in r.get('ahead', []) if e[0] == arena and
                ('span' not in r or r['region'] <= e[4] < r['region'] + r['span'])]

    def adaptive_counts(name, mode, windows, pages):
        r = runs.get((name, mode), {})
        before, after = r.get('ahead_before', {}), r.get('ahead_after', {})
        keys = ('fault_ahead_windows', 'fault_ahead_pages')
        if not instrumented or any(k not in before or k not in after for k in keys):
            return False
        delta = tuple(after[k] - before[k] for k in keys)
        events = r['ahead']
        own = region_events(r)
        other = [e for e in events if e not in own]
        return (delta == (len(events), sum(e[3] for e in events)) and
                (delta[0] - len(other), delta[1] - sum(e[3] for e in other)) == (windows, pages))

    # Fault indices: walk 0,1,2,7,16; stream adds 33,66,131,196,261,326,391,456.
    # An 8-page window that never doubles used to pass the broad fault ranges; it must fail.
    partly('adapt_walk_window', all(
        bool(got and base and got == (32, 5) and base[0] == 32)
        for got, base in zip(sources('adapt', 'walk'), sources('base1', 'walk'))) and adaptive_counts('adapt', 'walk', 3, 27))
    partly('adapt_stream_window', all(
        got == (512, 13) for got in sources('adapt', 'stream')) and adaptive_counts('adapt', 'stream', 11, 499))
    # The direct CV_STEP path may stop between batches. It must still request 4,8,16,32,64...
    # and continue at the first page NOT retained, while the exact sysfs deltas cover every
    # window (including zero-page cuts). A permanently small window fails the request sequence.
    for mode, size in (('walk', 32), ('stream', 512)):
        r = runs.get(('adaptdirect', mode), {})
        events = region_events(r)
        expected_index = 2
        growth = bool(events)
        for i, e in enumerate(events):
            want = min(4 << min(i, 4), 64)
            growth &= e[1] == expected_index and e[2] == want and 0 <= e[3] <= min(want, size - e[1] - 1)
            expected_index = e[1] + 1 + e[3]
        partly(f'adaptdirect_{mode}_growth', bool(
            growth and expected_index == size and sum(e[3] for e in events) > 0 and
            all(got == (size, len(events) + 2) for got in sources('adaptdirect', mode)) and
            adaptive_counts('adaptdirect', mode, len(events), size - len(events) - 2)))
    # A reschedule is raised after every demand GUP, so a window the step opens finds the stop
    # before its first batch: it pins nothing. The window still grows, one attempt per fault
    # (4, 8, 16, 32, 64) and the stream continues at the first page not retained. A fault the
    # launcher's CV_RESOLVE resolved first (a real stop landed between the fault and its
    # resolution in CV_STEP, by=resolve) gets the full window instead and pins up to want
    # pages: the walk then takes that many faults fewer. The route is read from the module's
    # own line for each window, so only those windows may pin, and never more than the
    # module's count of faults the launcher resolved first (launcher_first). That bound is the
    # report's, so without the report this check has no form that reads the kernel's records alone.
    r = runs.get(('adaptres', 'walk'), {})
    events = region_events(r)
    expected_index, growth = 2, bool(events)
    for i, e in enumerate(events):
        want = min(4 << min(i, 4), 64)
        growth &= (e[1] == expected_index and e[2] == want and 0 <= e[3] <= min(want, 31 - e[1]) and
                   (e[3] == 0 or e[5] == 'resolve'))
        expected_index = e[1] + 1 + e[3]
    pinned = sum(e[3] for e in events)
    graded('adaptres_walk_accounting', bool(
        growth and expected_index == 32 and
        sum(e[5] == 'resolve' for e in events) <= reports.get('adaptres', {}).get('launcher_first', 0) and
        all(got == (32, len(events) + 2) for got in sources('adaptres', 'walk')) and
        adaptive_counts('adaptres', 'walk', len(events), pinned)), counters and reported)
    for name, mode, accounting, events in (
            ('adaptgaps', 'gaps', (32, 7), [(3, 4, 3), (8, 8, 8), (17, 16, 14)]),
            ('adaptinterleave', 'interleave', (64, 10),
             # At 16 the first stream's window stops at its VMA's end; its next page then skips
             # the second array's pages already pinned, to 48, where the second stream's next
             # also is: the more recent first stream continues (window 32, clipped to 15).
             [(2, 4, 4), (34, 4, 4), (7, 8, 8), (39, 8, 8), (16, 16, 15), (48, 32, 15)])):
        actual = region_events(runs.get((name, mode), {}))
        partly(f'{name}_window', (
            [e[1:4] for e in actual] == events and
            all(got == accounting for got in sources(name, mode)) and
            adaptive_counts(name, mode, len(events), sum(e[2] for e in events))))
    # A short GUP retains half a batch, then the next fault must continue from the retained
    # end (not the requested end). Final fault at 31 has a trained but empty window. The
    # report's count of short results is a second witness, read where the log has a report.
    short = reports.get('adaptshort', {}).get('fault_short' if unpinned else 'short_results', 0) > 0
    partly('adaptshort_walk_window', (
        all(got == (32, 9) for got in sources('adaptshort', 'walk')) and
        adaptive_counts('adaptshort', 'walk', 7, 23) and (short or not reported)),
        counters and reported)
    # At page 7 the VMA boundary clips the entire window. Page 8 must continue the stream
    # with a 16-page window, and page 25 with 32 (clipped to six by the arena end).
    for name, mode in (('adaptclip', 'clip'), ('adaptclip', 'clipx'), ('adaptro', 'clipwrite')):
        r = runs.get((name, mode), {})
        checks[f'{name}_{mode}_window'] = (
            adaptive_counts(name, mode, 4, 26) and
            [e[1:4] for e in region_events(r)] == [(2, 4, 4), (7, 8, 0), (8, 16, 16), (25, 32, 6)])
        # Beside the read-only neighbour of 'clip' Linux refuses the write fault of an unclipped
        # window, which hides a missing VMA clip; the planned limit (not the resolved pages) shows it.
        if unpinned:
            checks[f'{name}_{mode}_window_in_vma'] = bool(r.get('region') is not None and
                unpinned_checks.windows_stay_below(r, r['region'] + 8 * 4096))
            checks[f'{name}_{mode}_window_in_arena'] = unpinned_checks.windows_in_arena(r)
        if name == 'adaptro':
            checks['adaptro_write_denied'] = r.get('rowrite', False) and r.get('exit') == 139
        else:
            partly(f'{name}_{mode}_accounting', all(got == (32, 6) for got in sources(name, mode)))
    # The launcher's grant line says whether the image's arena is non-linear (linear=0).
    r = runs.get(('adaptnonlin', 'nonlin'), {})
    partly('adaptnonlin_window', (
        (r.get('linear', {}).get(region_arena(r)) == 0 or not counters) and
        adaptive_counts('adaptnonlin', 'nonlin', 3, 27) and
        [((e[4] - r['region']) // 4096, e[2], e[3]) for e in region_events(r)] ==
        [(2, 4, 4), (7, 8, 8), (16, 16, 15)]))
    # Partial GUP results leave the rest to later faults: every page the walk
    # touched still ends pinned (at least window 1's pins), after more faults
    # than the full window takes. Unpinned, the same coverage is witnessed by Linux resolutions.
    partly('short64_walk_resolutions' if unpinned else 'short64_walk_pins', all(
        bool(got and low and full and got[0] >= low[0] and got[1] > full[1])
        for got, low, full in zip(sources('short64', 'walk'), sources('base1', 'walk'),
                                  sources('base64', 'walk'))))
    if unpinned:
        # The witness is only worth its zero if the counters count: the liveness launch, on
        # this kernel, acquired pins and released every one of them.
        live = ends.get(LIVE)
        checks['foll_pin_counters_live'] = bool(live and live[0] == live[1] > 0 and
            'HOOK_LIVE_STATUS:0' in log and 'HOOK_LIVE_MODE:Y' in log)
        for name in names:
            rep = reports.get(name, {})
            end = ends.get(name)
            first = runs.get((name, CONFIGS[name][1].split()[0]), {}).get('before')
            # The kernel's own FOLL_PIN counters read the same after the configuration's last
            # run as before its first: it acquired and released not one pin. The report adds
            # that the adapter ran unpinned and did resolve faults and enter the domain.
            partly(f'{name}_no_payload_pins', bool(end and first and end == first and (not reported or (
                rep.get('unpinned') == 1 and rep.get('fault_requests', 0) > 0 and rep.get('entries', 0) > 0))),
                reported)
            # A configuration that opens windows and injects no stop of its own must have opened
            # at least one batch (the OFF check required 0 < gup_max); one with an injected stop
            # may have cut every window before its first batch, so only the bound holds for it.
            params = CONFIGS[name][0]
            windowed = 'test_inject=' not in params and (
                name.startswith('adapt') or int(re.search(r'pin_ahead_pages=(\d+)', params + ' pin_ahead_pages=1')[1]) > 1)
            graded(f'{name}_fault_batch_bound', (rep.get('gup_requests') == 0 and
                rep.get('fault_requests', 0) > 0 and 0 <= rep.get('fault_max', -1) <= 16 and
                (not windowed or rep.get('fault_max', 0) > 0)), reported)
            graded(f'{name}_no_fault_while_pending', rep.get('fault_pending') == 0, reported)
        # A real quantum can cut any unpinned window, including CV_RESOLVE's.
        # Require the walk's exact page coverage and one event per demand
        # page, with public totals reconciled above. Each prefaulted page
        # must be witnessed; no allowance for unexplained extra events.
        for name in ('resched1', 'signal1', 'deadline1', 'resched2', 'signal2', 'deadline2',
                     'resched3', 'signal3', 'resched4', 'signal4', 'deadline4'):
            for mode in CONFIGS[name][1].split():
                r = runs.get((name, mode), {})
                counts = unpinned_checks.region_counts(r)
                arena = unpinned_checks.region_arena(r)
                events = [e for e in r.get('raw_events', []) if e['arena'] == arena]
                checks[f'{name}_{mode}_window_event_disjoint'] = unpinned_checks.disjoint(r) and bool(r.get('raw_events'))
                partly(f'{name}_{mode}_accounting', bool(counts and counts[0] == (32 if mode == 'walk' else 0) and
                    len({e['address'] // 4096 for e in events}) == len(events) and accounted(r)))
    return checks, runs, reports, ends, not_observable


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluate', type=Path, help='a saved serial.log: run the checks on it, no guest')
    p.add_argument('--instrumented', action='store_true', help='with --evaluate: the kernel had diag_admit')
    p.add_argument('--unpinned', action='store_true', help='require unpinned mode and its resolution oracle (the native adapter has no other mode)')
    p.add_argument('--qemu', type=Path)
    p.add_argument('--images', type=Path,
                   help="the production kit's images directory: accepted so that the gates share one "
                        'command line, and not booted here')
    p.add_argument('--adapter', type=Path, help='capstone-vexec from here')
    p.add_argument('--hooks-images', type=Path,
                   help='images directory of the hooks kit: the kernel Image built with CV_TEST_HOOKS=1, '
                        'with its fw_jump.elf and rootfs.ext2. Every boot of this gate uses it')
    p.add_argument('--application', type=Path, help='pin.dom')
    p.add_argument('--pthread-application', type=Path, help='pthread.dom; the liveness launch of the FOLL_PIN counters')
    p.add_argument('--malloc-application', type=Path, help='malloc.dom; without it the malloc configurations are skipped')
    p.add_argument('--work', type=Path)
    p.add_argument('--timeout', type=int, default=300, help='seconds for each boot (one per configuration)')
    p.add_argument('--module-args', default='',
                   help="adapter parameters added to every configuration's kernel command line, each as "
                        'capstone_vm.<parameter> (the broken-variant controls)')
    p.add_argument('--only', default='', help='comma-separated configuration names; the others are not run '
                   '(the checks of an unselected configuration then fail: read only the ones you name)')
    a = p.parse_args()
    if a.evaluate:
        log = a.evaluate.read_text(errors='replace').replace('\r', '')
        with_malloc = 'HOOK_PINS_BEFORE:malloc16:' in log
        names = [n for n in CONFIGS if app(n) == 'pin' or with_malloc]
        checks, _, _, _, not_observable = evaluate(log, names, a.instrumented, True, a.unpinned)
        report(checks, not_observable)
        return 0 if all(checks.values()) else 1
    for name in ('qemu', 'hooks_images', 'adapter', 'application', 'work'):
        if getattr(a, name) is None:
            p.error(f'--{name.replace("_", "-")} is required')
    if a.unpinned and not a.pthread_application:
        p.error('--unpinned requires --pthread-application')
    kernel = (a.hooks_images/'Image').read_bytes()
    if b'test_inject' not in kernel:
        p.error('--hooks-images: this kernel Image has no test hooks (make CV_TEST_HOOKS=1 Image)')
    instrumented = b'diag_admit' in kernel
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'; stage.mkdir()
    names = [n for n in CONFIGS if app(n) == 'pin' or a.malloc_application]
    if a.only:
        wanted = a.only.split(',')
        unknown = [n for n in wanted if n not in CONFIGS]
        if unknown:
            p.error('unknown configuration ' + ','.join(unknown))
        names = [n for n in names if n in wanted]
    inputs = {'qemu': a.qemu, 'application': a.application,
              'launcher': a.adapter/native_platform.LAUNCHER,
              **{n: a.hooks_images/n for n in ('Image', 'fw_jump.elf', 'rootfs.ext2')}}
    if a.malloc_application:
        inputs['malloc'] = a.malloc_application
    if a.unpinned:
        inputs['pthread'] = a.pthread_application
    hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    if a.malloc_application:
        shutil.copy2(a.malloc_application, stage/'malloc.dom')
    if a.unpinned:
        shutil.copy2(a.pthread_application, stage/'pthread.dom')
    for src, name in ((inputs['application'], 'pin.dom'),
                      (inputs['launcher'], native_platform.LAUNCHER)):
        shutil.copy2(src, stage/name)
    # One boot per configuration, and before them the liveness boot of an unpinned gate.
    boots = ([LIVE] if a.unpinned else []) + names
    gates, kernel_arguments, log, completed = {}, {}, '', True
    for name in boots:
        gates[name] = script(name)
        kernel_arguments[name] = parameters(name, instrumented, a.module_args)
        (stage/'gate.sh').write_text(gates[name])
        work = a.work/'guest'/name.lstrip('@')
        run = subprocess.run([os.environ.get('PYTHON', 'python3'),
            str(HERE/'run-staged.py'), '--qemu', str(a.qemu), '--images', str(a.hooks_images),
            '--stage', str(stage), '--work', str(work), '--disk-mib', '64',
            '--timeout', str(a.timeout),
            *(word for argument in kernel_arguments[name] for word in ('--kernel-arg', argument))])
        completed = completed and run.returncode == 0
        serial = work/'serial.log'
        log += serial.read_text(errors='replace').replace('\r', '') if serial.exists() else ''
    # The transcripts in boot order, as one file --evaluate can read.
    (a.work/'serial.log').write_text(log)
    checks, runs, reports, ends, not_observable = evaluate(log, names, instrumented, completed, a.unpinned)
    checks['inputs_unchanged'] = hashes == {
        k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    result = {'schema': 1, 'status': 'PASS' if all(checks.values()) else 'FAIL',
              'checks': checks, 'not_observable': not_observable,
              'instrumented': instrumented, 'unpinned': a.unpinned, 'module_args': a.module_args,
              'kernel_arguments': kernel_arguments,
              'reports': reports, 'pins_at_end': ends,
              'runs': {f'{n}:{m}': {k: (sorted(v) if isinstance(v, set) else v) for k, v in r.items()}
                       for (n, m), r in runs.items()}, 'sha256': hashes,
              'source_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), HERE/'pin-ahead-contract.c',
                                          HERE/'run-staged.py', HERE/'module/capstone_vm.c',
                                          HERE/'unpinned_checks.py']},
              'gate_sha256': hashlib.sha256(''.join(gates[name] for name in boots).encode()).hexdigest()}
    (a.work/'result.json').write_text(json.dumps(result, indent=2) + '\n')
    report(checks, not_observable)
    return 0 if all(checks.values()) else 1


def report(checks, not_observable):
    for name, ok in checks.items(): print(('PASS ' if ok else 'FAIL ') + name)
    for name in not_observable: print('NOT OBSERVABLE ' + name)


if __name__ == '__main__':
    raise SystemExit(main())
