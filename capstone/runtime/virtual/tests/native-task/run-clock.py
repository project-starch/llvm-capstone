#!/usr/bin/env python3
"""Qualify clock transport and output authority in the initial time namespace."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
for name in ['qemu', 'images', 'firmware', 'probes', 'out']:
    p.add_argument('--'+name, type=Path, required=True)
p.add_argument('--clock', type=int, choices=[0, 1], default=0)
p.add_argument('--fast', type=int, choices=[0, 1], default=1)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
stage = a.out/'stage'
stage.mkdir()
inputs = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
          [('qemu', a.qemu), ('Image', a.images/'Image'), ('firmware', a.firmware),
           ('rootfs', a.images/'rootfs.ext2')]}
for arm in ['exchange', 'direct', 'libc']:
    name = 'clock-'+arm+'.dom'
    shutil.copy2(a.probes/name, stage/name)
    inputs[name] = hashlib.sha256((stage/name).read_bytes()).hexdigest()
(stage/'gate.sh').write_text('''#!/bin/sh
set -eu
trap 'echo CLOCK_GATE_EXIT:$?; dmesg; echo VIRTUAL_STAGED_DONE' EXIT
cp -a /mnt/vm /tmp/clock-test
cd /tmp/clock-test
chmod +x *.dom
[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status
echo 0 > /proc/sys/kernel/randomize_va_space
for arm in exchange direct libc; do
 echo CLOCK_BEGIN:$arm
 ./clock-$arm.dom
 echo CLOCK_END:$arm
done
''')
configuration = dict(native_io=1, native_locks=1, native_stat=1, native_seek=1,
                     native_fd=1, native_close=1, native_path=1,
                     native_clock=a.clock, native_fast=a.fast)
flags = 'capstone_vm.native_task=1 capstone_vm.dom_binfmt=1 capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1 capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1 capstone_vm.supervisor_slots=32'
flags += ''.join(f' capstone_vm.{key}={value}' for key, value in configuration.items())
cmd = ['python3', str(Path(__file__).resolve().parents[2]/'run-staged.py'),
       '--qemu', str(a.qemu.resolve()), '--images', str(a.images.resolve()),
       '--firmware', str(a.firmware.resolve()), '--stage', str(stage.resolve()),
       '--work', str((a.out/'vm').resolve()), '--memory', '4G', '--disk-mib', '64',
       '--timeout', '180', '--icount', '0', '--kernel-arg', flags]
env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA='x-capstone-supervisor-slots=32,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true')
(a.out/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
with (a.out/'runner.log').open('w') as out:
    subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
log = (a.out/'vm/serial.log').read_text(errors='replace').replace('\r', '')
assert 'CLOCK_GATE_EXIT:0\n' in log and not re.search(r'WARNING:|BUG:|Oops|Kernel panic|FAIL', log)
rows = []
for arm in ['exchange', 'direct', 'libc']:
    body = log.split('CLOCK_BEGIN:'+arm+'\n')[1].split('CLOCK_END:'+arm+'\n')[0]
    checks = 30 if arm == 'direct' else 14
    assert re.findall(r'NATIVE_CLOCK checks=(\d+) failures=(\d+)', body) == [(str(checks), '0')]
    rows.append(dict(arm=arm, checks=checks, passed=True))
result = dict(status='PASS', inputs=inputs, configuration=configuration, rows=rows, limitations=[
    'Initial time namespace only; no time-namespace offset qualification.',
    'Only CLOCK_REALTIME and CLOCK_MONOTONIC are supported by this provider.',
    'Bad output authority takes precedence over an invalid clock ID in the inherited provider.',
    'Preflight rejection preserves output; no atomicity claim for faults during an actual partial copy.',
    'Native launch tick rate remains zero; legacy approximate clock synthesis is inactive.'])
(a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
