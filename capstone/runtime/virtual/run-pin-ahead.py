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
MODES = ('base', 'seq', 'stop', 'vma', 'rowrite', 'retire')
SCRIPT = '''#!/bin/sh
cd /mnt/vm || exit 1
export CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1
for w in ''' + ' '.join(map(str, WINDOWS)) + '''; do
  insmod capstone_vm.ko pin_ahead_pages=$w || exit 1
  for mode in ''' + ' '.join(MODES) + '''; do
    ./capstone-vexec pin.dom "$mode" > /tmp/pin.out 2>&1
    echo PIN_EXIT:$w:$mode:$?
    cat /tmp/pin.out
  done
  ./capstone-vexec --stats
  rmmod capstone_vm || exit 1
done
echo VIRTUAL_STAGED_DONE
'''
# Pins the mode adds over "base" (pinned at exit, peak, fault events) for a
# window of w pages over the 32-page region; see pin-ahead-contract.c. For
# retire the peak is not additive (base reaches its own peak after the
# munmap), so only the pins left at exit and the one fault are compared; the
# window's size there is the one seq shows.
EXPECT = {
    'seq':    lambda w: (min(w, 32), min(w, 32), 1),
    'stop':   lambda w: (min(w, 28) + min(w, 4), min(w, 28) + min(w, 4), 2),
    'vma':    lambda w: (min(w, 8) + min(w, 24), min(w, 8) + min(w, 24), 2),
    'retire': lambda w: (0, None, 1),
}


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
    (stage/'gate.sh').write_text(SCRIPT)
    run = subprocess.run([os.environ.get('PYTHON', 'python3'),
        str(HERE/'run-staged.py'), '--qemu', str(a.qemu), '--images', str(a.images),
        '--stage', str(stage), '--work', str(a.work/'guest'), '--disk-mib', '64',
        '--timeout', str(a.timeout)])
    log = (a.work/'guest/serial.log').read_text(errors='replace').replace('\r', '')
    # Each PIN_EXIT line is followed by that run's output, its stats included.
    runs, current = {}, None
    for line in log.splitlines():
        m = re.match(r'PIN_EXIT:(\d+):(\w+):(\d+)$', line)
        if m:
            current = (int(m.group(1)), m.group(2))
            runs[current] = {'exit': int(m.group(3)), 'stats': None, 'ok': False}
        elif current and line.startswith('CAPSTONE_VM_STATS '):
            runs[current]['stats'] = dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
        elif current and line == 'PIN_OK:' + current[1]:
            runs[current]['ok'] = True
    census = [json.loads(l) for l in log.splitlines() if l.startswith('{"version":1,')]
    checks = {'completed': run.returncode == 0,
              'all_runs': len(runs) == len(WINDOWS) * len(MODES),
              'no_contract_failure': 'PIN_FAIL:' not in log}
    deltas = {}
    for w in WINDOWS:
        base = runs.get((w, 'base'))
        checks[f'w{w}_base'] = bool(base and base['exit'] == 0 and base['ok'] and base['stats'])
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
              'checks': checks, 'deltas_pages_peak_faults': deltas,
              'runs': {f'{w}:{m}': v for (w, m), v in runs.items()},
              'sha256': hashes,
              'source_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), HERE/'pin-ahead-contract.c',
                                          HERE/'run-staged.py']},
              'gate_sha256': hashlib.sha256(SCRIPT.encode()).hexdigest(),
              'scope': 'one hart, private anonymous arenas, windows ' + ', '.join(map(str, WINDOWS))}
    (a.work/'result.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, ok in checks.items(): print(('PASS ' if ok else 'FAIL ') + name)
    print('deltas (pages, peak, faults):', json.dumps(deltas))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
