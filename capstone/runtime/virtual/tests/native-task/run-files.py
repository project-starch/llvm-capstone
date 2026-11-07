#!/usr/bin/env python3
"""Real Linux contention and native file-service qualification in a fresh guest."""
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
p.add_argument('--locks', type=int, choices=[0, 1], default=0)
p.add_argument('--fast', type=int, choices=[0, 1], default=1)
p.add_argument('--stat', type=int, choices=[0, 1], default=0)
p.add_argument('--seek', type=int, choices=[0, 1], default=0)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
stage = a.out/'stage'
stage.mkdir()
inputs = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
          [('qemu', a.qemu), ('Image', a.images/'Image'), ('firmware', a.firmware),
           ('rootfs', a.images/'rootfs.ext2')]}
for name in ['files-exchange.dom', 'files-direct.dom', 'files-native.dom', 'files-driver']:
    shutil.copy2(a.probes/name, stage/name)
    inputs[name] = hashlib.sha256((stage/name).read_bytes()).hexdigest()
(stage/'gate.sh').write_text('''#!/bin/sh
set -eu
trap 'echo FILES_GATE_EXIT:$?; dmesg; echo VIRTUAL_STAGED_DONE' EXIT
cp -a /mnt/vm /tmp/files-test
cd /tmp/files-test
chmod +x files-driver *.dom
[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status
echo 0 > /proc/sys/kernel/randomize_va_space
printf sentinel > /tmp/outside-sentinel
ln -s /tmp escape
for arm in exchange direct native; do
 echo FILES_BEGIN:$arm
 ./files-driver ./files-$arm.dom
 test "$(cat /tmp/outside-sentinel)" = sentinel
 echo FILES_END:$arm
done
''')
flags = 'capstone_vm.native_task=1 capstone_vm.native_io=1 capstone_vm.dom_binfmt=1 capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1 capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1 capstone_vm.supervisor_slots=32'
flags += f' capstone_vm.native_locks={a.locks} capstone_vm.native_stat={a.stat} capstone_vm.native_seek={a.seek} capstone_vm.native_fast={a.fast}'
cmd = ['python3', str(Path(__file__).resolve().parents[2]/'run-staged.py'),
       '--qemu', str(a.qemu.resolve()), '--images', str(a.images.resolve()),
       '--firmware', str(a.firmware.resolve()), '--stage', str(stage.resolve()),
       '--work', str((a.out/'vm').resolve()), '--memory', '4G', '--disk-mib', '64',
       '--timeout', '180', '--icount', '0', '--kernel-arg', flags]
env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA='x-capstone-supervisor-slots=32,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true')
(a.out/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
with (a.out/'runner.log').open('w') as out:
    subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
log = (a.out/'vm/serial.log').read_text(errors='replace')
assert 'FILES_GATE_EXIT:0\n' in log and not re.search(r'WARNING:|BUG:|Oops|Kernel panic|_FAIL', log)
rows = []
for arm in ['exchange', 'direct', 'native']:
    body = log.split('FILES_BEGIN:'+arm+'\n')[1].split('FILES_END:'+arm+'\n')[0]
    expected = {'contend': 30 if arm == 'direct' else 22, 'wait': 7, 'basic': 76 if arm == 'direct' else 58}
    for mode, checks in expected.items():
        assert f'NATIVE_FILES mode={mode} checks={checks} failures=0\n' in body
    assert 'NATIVE_FILES_DRIVER checks=56 failures=0\n' in body
    rows.append(dict(arm=arm, domain_checks=expected, linux_checks=56, passed=True))
result = dict(inputs=inputs, configuration=dict(native_locks=a.locks, native_stat=a.stat, native_seek=a.seek, native_fast=a.fast), rows=rows, limitations=[
    'Single-hart root-only prototype, default-off task backend; not all Linux syscalls.',
    'Real Linux POSIX locks and process lifecycle; no caught-signal/restart or multithreaded-domain claim.',
    'tmpfs file correctness and sync error checks, not a crash-durability test.'])
(a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
