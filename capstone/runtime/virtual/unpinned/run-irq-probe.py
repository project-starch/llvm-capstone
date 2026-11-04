#!/usr/bin/env python3
"""Run the firmware-timer window fixture in a private, queued guest and report
its counters. The caller compares counters between kernels and CPU variants."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qemu', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--bulk', choices=['on', 'off'], default='off')
    parser.add_argument('--cpu-extra', default='')
    parser.add_argument('--icount', action='store_true',
                        help='instruction-exact virtual time (-icount shift=0), as the census platforms run')
    parser.add_argument('--params', default='iters=4000 dmin=1 dmax=24')
    parser.add_argument('--expect-lost', choices=['any', 'none', 'record'], default='record')
    parser.add_argument('--expect-retries', choices=['any', 'none', 'record'], default='record')
    parser.add_argument('--expect-window', choices=['inside', 'outside', 'record'], default='record',
                        help='whether firmware timer deadlines fall inside the copy call')
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    stage = args.work / 'stage'
    stage.mkdir()
    shutil.copy2(args.module, stage / 'irq-window-probe.ko')
    (stage / 'gate.sh').write_text('''#!/bin/sh
cd /mnt/vm || exit 1
echo BULK_ENABLED:$(cat /sys/module/captag/parameters/bulk_enabled)
insmod irq-window-probe.ko %s
echo PROBE_INSMOD_RC:$?
dmesg | grep IRQ_WINDOW
for f in copies copy_retries; do echo CAPTAG_$f:$(cat /sys/module/captag/parameters/$f 2>/dev/null); done
dmesg | grep -i 'sstc'
echo VIRTUAL_STAGED_DONE
''' % args.params)
    here = Path(__file__).resolve().parent
    runner = here.parent / 'run-staged.py'
    if args.icount:
        # Same runner with the census platforms' instruction-exact clock added.
        text = runner.read_text()
        marker = 'completed = login = sent = False'
        assert text.count(marker) == 1
        runner = args.work / 'run-staged-icount.py'
        runner.write_text(text.replace(marker, "cmd += ['-icount', 'shift=0']\n    " + marker))
    cmd = [sys.executable, str(runner),
           '--qemu', str(args.qemu.resolve()), '--images', str(args.images.resolve()),
           '--stage', str(stage.resolve()), '--work', str((args.work / 'guest').resolve()),
           '--disk-mib', '16', '--timeout', '900', '--exact-bounds',
           '--kernel-arg', 'captag_bulk=' + args.bulk,
           '--kernel-arg', 'init_on_alloc=0', '--kernel-arg', 'init_on_free=0']
    paths = [args.qemu, args.module] + [args.images / n for n in ['Image', 'fw_jump.elf', 'rootfs.ext2']]
    paths += [Path(__file__), runner, here / 'linux-6.1-unpinned.patch']
    extra = 'x-capstone-raw-moves=true,x-capstone-test-csrs=false'
    if args.cpu_extra:
        extra += ',' + args.cpu_extra
    record = {'icount': args.icount, 'command': cmd, 'cpu_extra': extra, 'params': args.params,
              'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.work / 'inputs.json').write_text(json.dumps(record, indent=2) + '\n')
    env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA=extra,
               PYTHONPATH=os.pathsep.join([str(here.parent), os.environ.get('PYTHONPATH', '')]))
    with (args.work / 'runner.log').open('w') as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    serial_path = args.work / 'guest/serial.log'
    serial = serial_path.read_text(errors='replace') if serial_path.exists() else ''
    summary = re.search(r'IRQ_WINDOW iterations=(\d+) deadline_in_copy=(\d+) lost_iterations=(\d+) '
                        r'lost_in_copy=(\d+) lost_granules=(\d+)', serial)
    retries = re.search(r'CAPTAG_copy_retries:(\d+)', serial)
    copies = re.search(r'CAPTAG_copies:(\d+)', serial)
    cpu_txt = args.work / 'guest/cpu.txt'
    cpu = cpu_txt.read_text() if cpu_txt.exists() else ''
    # The requested configuration must be visible in the artifacts, or the run says nothing.
    config_ok = ('BULK_ENABLED:' + ('Y' if args.bulk == 'on' else 'N') in serial and
                 all(prop in cpu for prop in extra.split(',')))
    ok = (result.returncode == 0 and 'IRQ_WINDOW DONE' in serial and
          'VIRTUAL_STAGED_DONE' in serial and 'thread create refused' not in serial and
          config_ok and bool(summary) and int(summary.group(1)) > 0)
    iterations = int(summary.group(1)) if summary else -1
    inside = int(summary.group(2)) if summary else -1
    lost = int(summary.group(3)) if summary else -1
    redo = int(retries.group(1)) if retries else -1
    if args.expect_lost != 'record':
        ok = ok and ((lost > 0) if args.expect_lost == 'any' else (lost == 0))
    if args.expect_retries != 'record':
        ok = ok and ((redo > 0) if args.expect_retries == 'any' else (redo == 0))
    if args.expect_window != 'record':
        ok = ok and ((inside > 0) if args.expect_window == 'inside' else (inside == 0))
    answer = {'pass': ok, 'runner_rc': result.returncode, 'config_ok': config_ok,
              'bulk': args.bulk, 'cpu_extra': extra, 'iterations': iterations,
              'deadline_in_copy': inside, 'lost_iterations': lost, 'copy_retries': redo,
              'copies': int(copies.group(1)) if copies else -1,
              'expect': {'lost': args.expect_lost, 'retries': args.expect_retries,
                         'window': args.expect_window}}
    (args.work / 'result.json').write_text(json.dumps(answer, indent=2) + '\n')
    print(json.dumps(answer))
    for line in serial.splitlines():
        if 'IRQ_WINDOW' in line or 'BULK_ENABLED' in line or 'CAPTAG_' in line or 'sstc' in line.lower():
            print(line)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
