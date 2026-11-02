#!/usr/bin/env python3
"""The adapter's pin-ahead window: forward clip to the arena, stop at a pinned
page and at the VMA end, read-only (FOLL_FORCE) backing stays read-only, and
retirement releases every prefetched pin. Window 1 is the control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
WINDOWS = (1, 16, 64)
MODES = ('base', 'seq', 'stop', 'vma', 'vmax', 'rowrite', 'retire')
# With a module that has the diag_admit parameter, every page an arena gains or loses is one
# dmesg line (admit, window, scan, touch, retire ... swept N), read after each run with the
# ring buffer cleared, so the checks below count the pins of the REGION'S arena alone. The
# console is quieted first (dmesg -n 1): a window's lines written to the serial console from
# inside the window cost more than the step's quantum and cut the window's second batch. The
# process-wide total also holds the program's own pages, which since lazy NONLIN grants are
# admitted at first touch and vary with placement (the allocator's address-hashed index in
# .bss) and with windows cut by the quantum; a module without the parameter (the control
# before that change) is measured on the process total, as before.
SCRIPT = '''#!/bin/sh
cd /mnt/vm || exit 1
export CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1
dmesg -n 1
dmesg -c > /dev/null
for w in ''' + ' '.join(map(str, WINDOWS)) + '''; do
  insmod capstone_vm.ko pin_ahead_pages=$w @DIAG_PARAM@ || exit 1
  for mode in ''' + ' '.join(MODES) + '''; do
    ./capstone-vexec pin.dom "$mode" > /tmp/pin.out 2>&1
    echo PIN_EXIT:$w:$mode:$?
    cat /tmp/pin.out
    dmesg -c | grep "capstone_vm: " | sed "s/^/PIN_DIAG:/"
  done
  ./capstone-vexec --stats
  rmmod capstone_vm || exit 1
done
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
# Uninstrumented module: pins the mode adds over "base" (pinned at exit,
# peak, fault events), process-wide. For retire the peak is not additive
# (base reaches its own peak after the munmap), so only the pins left at exit
# and the one fault are compared; the window's size there is the one seq shows.
EXPECT = {
    'seq':    lambda w: (min(w, 32), min(w, 32), 1),
    'stop':   lambda w: (min(w, 28) + min(w, 4), min(w, 28) + min(w, 4), 2),
    'vma':    lambda w: (min(w, 8) + min(w, 24), min(w, 8) + min(w, 24), 2),
    'vmax':   lambda w: (min(w, 8), min(w, 8), 1),
    'retire': lambda w: (0, None, 1),
}


def region_counts(run):
    """(pins, faults, swept) of the arena holding the run's region, from its diag lines."""
    arena = next((base for base, bytes_ in run['arenas']
                  if base <= run['region'] < base + bytes_), None) if run['region'] is not None else None
    if arena is None:
        return None
    pins = faults = swept = 0
    for kind, base, value in run['diag']:
        if base != arena:
            continue
        if kind in ('admit', 'window', 'scan', 'touch'):
            pins += 1
        if kind == 'admit':
            faults += 1
        if kind == 'retire':
            swept += value
    return pins, faults, swept


DIAG = re.compile(r'capstone_vm: (admit|window|scan|touch|retire) arena 0x([0-9a-f]+)\+0x[0-9a-f]+ '
                  r'(?:page (\d+)|swept (\d+))')
ADD_OK = re.compile(r'CAPSTONE_VM_ADD_OK slot=\d+ base=0x([0-9a-f]+) bytes=(\d+) ')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qemu', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--application', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'; stage.mkdir()
    inputs = {'qemu': a.qemu, 'application': a.application,
              'launcher': a.adapter/'capstone-vexec',
              'module': a.adapter/'module/capstone_vm.ko',
              **{n: a.images/n for n in ('Image', 'fw_jump.elf', 'rootfs.ext2')}}
    hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    for src, name in ((inputs['application'], 'pin.dom'),
                      (inputs['launcher'], 'capstone-vexec'),
                      (inputs['module'], 'capstone_vm.ko')):
        shutil.copy2(src, stage/name)
    instrumented = b'diag_admit' in inputs['module'].read_bytes()
    gate = SCRIPT.replace('@DIAG_PARAM@', 'diag_admit=1' if instrumented else '')
    (stage/'gate.sh').write_text(gate)
    run = subprocess.run([os.environ.get('PYTHON', 'python3'),
        str(HERE/'run-staged.py'), '--qemu', str(a.qemu), '--images', str(a.images),
        '--stage', str(stage), '--work', str(a.work/'guest'), '--disk-mib', '64',
        '--timeout', str(a.timeout)])
    log = (a.work/'guest/serial.log').read_text(errors='replace').replace('\r', '')
    # Each PIN_EXIT line is followed by that run's output (its stats and region included) and
    # then the module's diag lines of that run.
    runs, current = {}, None
    for line in log.splitlines():
        m = re.match(r'PIN_EXIT:(\d+):(\w+):(\d+)$', line)
        if m:
            current = (int(m.group(1)), m.group(2))
            runs[current] = {'exit': int(m.group(3)), 'stats': None, 'ok': False,
                             'region': None, 'arenas': [], 'diag': []}
        elif current and line.startswith('CAPSTONE_VM_STATS '):
            runs[current]['stats'] = dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
        elif current and line == 'PIN_OK:' + current[1]:
            runs[current]['ok'] = True
        elif current and line.startswith('PIN_REGION:0x'):
            runs[current]['region'] = int(line[len('PIN_REGION:'):], 16)
        elif current and (m := ADD_OK.search(line)):
            runs[current]['arenas'].append((int(m.group(1), 16), int(m.group(2))))
        elif current and line.startswith('PIN_DIAG:') and (m := DIAG.search(line)):
            runs[current]['diag'].append((m.group(1), int(m.group(2), 16),
                                          int(m.group(3) or m.group(4))))
    census = [json.loads(l) for l in log.splitlines() if l.startswith('{"version":1,')]
    checks = {'completed': run.returncode == 0,
              'all_runs': len(runs) == len(WINDOWS) * len(MODES),
              'no_contract_failure': 'PIN_FAIL:' not in log}
    deltas = {}
    for w in WINDOWS:
        base = runs.get((w, 'base'))
        checks[f'w{w}_base'] = bool(base and base['exit'] == 0 and base['ok'] and base['stats'])
        if instrumented:
            for mode, expect in EXPECT_REGION.items():
                r = runs.get((w, mode))
                got = region_counts(r) if r else None
                ok = bool(r and r['exit'] == 0 and r['ok'] and got is not None)
                if ok:
                    deltas[f'w{w}_{mode}'] = got
                    ok = got == expect(w)
                checks[f'w{w}_{mode}'] = ok
        else:
            for mode, expect in EXPECT.items():
                r = runs.get((w, mode))
                ok = bool(r and r['exit'] == 0 and r['ok'] and r['stats'] and checks[f'w{w}_base'])
                if ok:
                    s, b = r['stats'], base['stats']
                    got = (s['pages'] - b['pages'], s['peak'] - b['peak'], s['faults'] - b['faults'])
                    deltas[f'w{w}_{mode}'] = got
                    ok = all(e is None or g == e for g, e in zip(got, expect(w)))
                checks[f'w{w}_{mode}'] = ok
        r = runs.get((w, 'rowrite'))
        checks[f'w{w}_rowrite_faults'] = bool(r and r['exit'] == 139 and not r['ok'])
    checks['rowrite_reached_store'] = log.count('PIN_ROWRITE_ATTEMPT') == len(WINDOWS)
    checks['denials_are_store_faults'] = re.findall(
        r'^capstone-exec: domain fault cause=(\d+) ', log, re.M) == ['15'] * len(WINDOWS)
    checks['cleanup'] = len(census) == len(WINDOWS) and all(
        s['live_domains'] == s['live_regions'] == s['live_bytes'] == 0 for s in census)
    checks['inputs_unchanged'] = hashes == {
        k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    result = {'schema': 1, 'status': 'PASS' if all(checks.values()) else 'FAIL',
              'checks': checks, 'instrumented': instrumented,
              ('region_pins_faults_swept' if instrumented else 'deltas_pages_peak_faults'): deltas,
              'runs': {f'{w}:{m}': v for (w, m), v in runs.items()},
              'sha256': hashes,
              'source_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), HERE/'pin-ahead-contract.c',
                                          HERE/'run-staged.py']},
              'gate_sha256': hashlib.sha256(gate.encode()).hexdigest(),
              'scope': 'one hart, private anonymous arenas, windows ' + ', '.join(map(str, WINDOWS))}
    (a.work/'result.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, ok in checks.items(): print(('PASS ' if ok else 'FAIL ') + name)
    print(('region (pins, faults, swept):' if instrumented else 'deltas (pages, peak, faults):'),
          json.dumps(deltas))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
