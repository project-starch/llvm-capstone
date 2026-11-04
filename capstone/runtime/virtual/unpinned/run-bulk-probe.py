#!/usr/bin/env python3
"""Run the dense page-copy/zero/reuse fixture in a private, queued guest."""
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
    parser.add_argument('--bulk', choices=['on', 'off'], default='on')
    parser.add_argument('--kernel-arg', action='append', default=[])
    parser.add_argument('--expect', choices=['pass', 'lost-copy', 'stale-free', 'leaked-tag'], default='pass')
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    stage = args.work / 'stage'
    stage.mkdir()
    shutil.copy2(args.module, stage / 'bulk-probe.ko')
    (stage / 'gate.sh').write_text('''#!/bin/sh
cd /mnt/vm || exit 1
cat /proc/cmdline
echo BULK_ENABLED:$(cat /sys/module/captag/parameters/bulk_enabled)
insmod bulk-probe.ko
rc=$?
echo BULK_INSMOD_RC:$rc
[ "$rc" != 0 ] || rmmod bulk_probe
dmesg | grep BULK_MM
echo VIRTUAL_STAGED_DONE
''')
    here = Path(__file__).resolve().parent
    cmd = [sys.executable, str(here.parent / 'run-staged.py'),
           '--qemu', str(args.qemu.resolve()), '--images', str(args.images.resolve()),
           '--stage', str(stage.resolve()), '--work', str((args.work / 'guest').resolve()),
           '--disk-mib', '16', '--timeout', '180', '--exact-bounds',
           '--kernel-arg', 'captag_bulk=' + args.bulk,
           '--kernel-arg', 'init_on_alloc=0', '--kernel-arg', 'init_on_free=0']
    for arg in args.kernel_arg:
        cmd += ['--kernel-arg', arg]
    paths = [args.qemu, args.module] + [args.images / n for n in ['Image', 'fw_jump.elf', 'rootfs.ext2']]
    paths += [Path(__file__), here.parent / 'run-staged.py', here / 'linux-6.1-unpinned.patch']
    record = {'command': cmd, 'expected': args.expect,
              'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.work / 'inputs.json').write_text(json.dumps(record, indent=2) + '\n')
    env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA='x-capstone-raw-moves=true,x-capstone-test-csrs=false')
    with (args.work / 'runner.log').open('w') as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    serial_path = args.work / 'guest/serial.log'
    serial = serial_path.read_text(errors='replace') if serial_path.exists() else ''
    # Never accept an early boot failure as a negative-control success.
    enabled = 'BULK_ENABLED:' + ('Y' if args.bulk == 'on' else 'N') in serial
    if args.expect == 'pass':
        ok = (result.returncode == 0 and 'BULK_MM PASS' in serial and
              'BULK_INSMOD_RC:0' in serial and
              serial.count('BULK_MM copy missing_tags=0 extra_tags=0 bad_words=0 granules=256') >= 2 and
              serial.count('BULK_MM mixed_copy missing_tags=0 extra_tags=0 bad_words=0 granules=256') >= 2 and
              'zero page=0 sticky=1 actual_tags=0' in serial and
              'zero page=1 sticky=1 actual_tags=0' in serial and
              'kernel_reuse same_frame=1 actual_tags=0 pages=2' in serial)
    elif args.expect == 'lost-copy':
        ok = (result.returncode == 0 and 'BULK_MM FAIL' in serial and
              bool(re.search(r'copy missing_tags=[1-9][0-9]* ', serial)) and
              bool(re.search(r'BULK_INSMOD_RC:[1-9]', serial)))
    elif args.expect == 'stale-free':
        # A BUG may kill insmod while the guest shell remains usable.
        ok = ((result.returncode != 0 or 'BULK_INSMOD_RC:139' in serial) and
              'kernel BUG at' in serial and
              'arch_free_page' in serial and
              'zero page=1 sticky=1 actual_tags=0' in serial)
    else:
        ok = (result.returncode == 0 and 'BULK_MM FAIL' in serial and
              bool(re.search(r'kernel_reuse same_frame=1 actual_tags=[1-9]', serial)) and
              bool(re.search(r'BULK_INSMOD_RC:[1-9]', serial)))
    if args.expect != 'stale-free':
        ok = ok and 'VIRTUAL_STAGED_DONE' in serial
    ok = enabled and ok
    if 'thread create refused' in serial or 'Exec format error' in serial:
        ok = False
    answer = {'pass': ok, 'expected': args.expect, 'runner_rc': result.returncode,
              'bulk_readback': enabled}
    (args.work / 'result.json').write_text(json.dumps(answer, indent=2) + '\n')
    print(json.dumps(answer))
    for line in serial.splitlines():
        if line.startswith('[') and 'BULK_MM' in line:
            print(line)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
