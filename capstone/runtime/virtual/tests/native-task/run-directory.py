#!/usr/bin/env python3
"""Qualify confined directory services in both native syscall ABI variants."""
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
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--runner', type=Path,
               default=Path(__file__).resolve().parents[2] / 'run-staged.py')
p.add_argument('--time-probes', type=Path,
               help='Optional directory containing time-services-exchange.dom and time-services-direct.dom')
p.add_argument('--dup-probes', type=Path,
               help='Optional directory containing dup-services-exchange.dom and dup-services-direct.dom')
a = p.parse_args()
if not os.environ.get('CAPSTONE_QEMU_LOCK'):
    p.error('CAPSTONE_QEMU_LOCK is required; source capstone/tests/capstone-test-env.sh first')
a.out.mkdir(parents=True, exist_ok=False)
stage = a.out / 'stage'
stage.mkdir()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

inputs = {name: dict(path=str(path.resolve()), sha256=digest(path)) for name, path in [
    ('qemu', a.qemu), ('Image', a.images / 'Image'),
    ('firmware', a.firmware), ('rootfs', a.images / 'rootfs.ext2')]}
for arm in ['exchange', 'direct']:
    name = 'directory-services-' + arm + '.dom'
    shutil.copy2(a.probes / name, stage / name)
    inputs[name] = dict(path=str((a.probes / name).resolve()), sha256=digest(stage / name))
    if a.time_probes:
        name = 'time-services-' + arm + '.dom'
        shutil.copy2(a.time_probes / name, stage / name)
        inputs[name] = dict(path=str((a.time_probes / name).resolve()), sha256=digest(stage / name))
    if a.dup_probes:
        name = 'dup-services-' + arm + '.dom'
        shutil.copy2(a.dup_probes / name, stage / name)
        inputs[name] = dict(path=str((a.dup_probes / name).resolve()), sha256=digest(stage / name))

gate = '''#!/bin/sh
set -eu
trap 'echo DIRECTORY_GATE_EXIT:$?; dmesg; echo VIRTUAL_STAGED_DONE' EXIT
[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status
echo 0 > /proc/sys/kernel/randomize_va_space
mkdir -p /tmp/outside
printf sentinel > /tmp/outside/marker
umask 0027
for arm in exchange direct; do
 root=/tmp/directory-root-$arm
 mkdir -p "$root"
 cp /mnt/vm/directory-services-$arm.dom "$root/probe.dom"
 chmod +x "$root/probe.dom"
 cd "$root"
 echo DIRECTORY_BEGIN:$arm
 ./probe.dom
 test "$(cat /tmp/outside/marker)" = sentinel
 test ! -e /tmp/outside/nope
 test ! -e /tmp/outside/hard-escape
 test ! -e dir
 echo DIRECTORY_END:$arm
done
'''
if a.dup_probes:
    gate += '''for arm in exchange direct; do
 root=/tmp/dup-root-$arm
 mkdir -p "$root"
 cp /mnt/vm/dup-services-$arm.dom "$root/probe.dom"
 chmod +x "$root/probe.dom"
 cd "$root"
 echo DUP_BEGIN:$arm
 ./probe.dom
 echo DUP_END:$arm
done
'''
if a.time_probes:
    gate += '''for arm in exchange direct; do
 root=/tmp/time-root-$arm
 mkdir -p "$root"
 cp /mnt/vm/time-services-$arm.dom "$root/probe.dom"
 chmod +x "$root/probe.dom"
 cd "$root"
 echo TIME_BEGIN:$arm
 ./probe.dom
 echo TIME_END:$arm
done
'''
(stage / 'gate.sh').write_text(gate)
(stage / 'gate.sh').chmod(0o755)
flags = ' '.join('capstone_vm.' + item for item in [
    'native_task=1', 'native_io=1', 'dom_binfmt=1', 'native_dispatch=1',
    'checked_copy=1', 'native_slot_transport=1', 'unpinned=1',
    'supervisor_slots=1', 'native_refused_trace=1'])
cmd = ['python3', str(a.runner.resolve()),
       '--qemu', str(a.qemu.resolve()), '--images', str(a.images.resolve()),
       '--firmware', str(a.firmware.resolve()), '--stage', str(stage.resolve()),
       '--work', str((a.out / 'vm').resolve()), '--memory', '8G',
       '--disk-mib', '64', '--timeout', '240', '--icount', '0', '--kernel-arg', flags]
cpu = ','.join(['x-capstone-supervisor-slots=1', 'x-pcrel=false',
                'x-capstone-raw-moves=true', 'x-capstone-test-csrs=false',
                'x-capstone-checked-copy=true', 'x-capstone-context-transport=true'])
env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA=cpu)
(a.out / 'command.json').write_text(json.dumps(dict(argv=cmd, cpu_extra=cpu), indent=2) + '\n')
(a.out / 'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
with (a.out / 'runner.log').open('w') as out:
    subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
log = (a.out / 'vm/serial.log').read_text(errors='replace')
assert 'DIRECTORY_GATE_EXIT:0\n' in log
assert not re.search(r'WARNING:|BUG:|Oops|Kernel panic|_FAIL', log)
rows = []
time_rows = []
dup_rows = []
for arm in ['exchange', 'direct']:
    body = log.split('DIRECTORY_BEGIN:' + arm + '\n')[1].split('DIRECTORY_END:' + arm + '\n')[0]
    match = re.search(r'NATIVE_DIRECTORY checks=(\d+) failures=0\n', body)
    assert match and int(match[1]) >= 45
    rows.append(dict(arm=arm, checks=int(match[1]), failures=0))
    if a.time_probes:
        body = log.split('TIME_BEGIN:' + arm + '\n')[1].split('TIME_END:' + arm + '\n')[0]
        match = re.search(r'TIME checks=(\d+) failures=0\n', body)
        assert match and int(match[1]) > 0
        time_rows.append(dict(arm=arm, checks=int(match[1]), failures=0))
    if a.dup_probes:
        body = log.split('DUP_BEGIN:' + arm + '\n')[1].split('DUP_END:' + arm + '\n')[0]
        match = re.search(r'NATIVE_DUP checks=(\d+) failures=0\n', body)
        assert match and int(match[1]) > 0
        dup_rows.append(dict(arm=arm, checks=int(match[1]), failures=0))
result = dict(status='PASS', inputs=inputs, rows=rows, time_rows=time_rows, dup_rows=dup_rows,
              limitations=['Single-hart root-only native task prototype.',
                           'Symlinks are inspected, never followed for data or directory access.',
                           'No rename-race, mount-race, caught-signal or generic-fork claim.'])
(a.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
