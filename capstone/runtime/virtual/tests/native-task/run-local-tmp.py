#!/usr/bin/env python3
"""Qualify the opt-in local /tmp alias, inheritance and confinement controls."""
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
a = p.parse_args()
if not os.environ.get('CAPSTONE_QEMU_LOCK'):
    p.error('CAPSTONE_QEMU_LOCK is required; source capstone/tests/capstone-test-env.sh first')
a.out.mkdir(parents=True, exist_ok=False)
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
inputs = {name: {'path': str(path.resolve()), 'sha256': digest(path)} for name, path in [
    ('qemu', a.qemu), ('Image', a.images / 'Image'),
    ('firmware', a.firmware), ('rootfs', a.images / 'rootfs.ext2')]}
rows = []
for mode in ['off', 'on', 'on-null', 'conflict']:
    work = a.out / mode
    stage = work / 'stage'
    stage.mkdir(parents=True)
    for name in ['local-tmp-exchange.dom', 'local-tmp-direct.dom', 'local-tmp-driver']:
        shutil.copy2(a.probes / name, stage / name)
        (stage / name).chmod(0o755)
        inputs[name] = {'path': str((a.probes / name).resolve()), 'sha256': digest(stage / name)}
    gate = '''#!/bin/sh
set -eu
trap 'echo LOCAL_TMP_GATE_EXIT:MODE:$?; dmesg; echo VIRTUAL_STAGED_DONE' EXIT
[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status
echo 0 > /proc/sys/kernel/randomize_va_space
cp -a /mnt/vm /tmp/local-tmp-probes-MODE
cd /tmp/local-tmp-probes-MODE
chmod +x local-tmp-driver *.dom
for arm in exchange direct; do
 echo LOCAL_TMP_BEGIN:MODE:$arm
 ./local-tmp-driver "$PWD/local-tmp-$arm.dom" MODE
 echo LOCAL_TMP_END:MODE:$arm
done
'''.replace('MODE', mode)
    (stage / 'gate.sh').write_text(gate)
    (stage / 'gate.sh').chmod(0o755)
    flags = ['native_task=1', 'native_io=1', 'dom_binfmt=1', 'native_dispatch=1',
             'checked_copy=1', 'native_slot_transport=1', 'unpinned=1', 'supervisor_slots=1']
    if mode != 'off': flags.append('dom_local_tmp=1')
    if mode == 'on-null': flags.append('dom_null=1')
    if mode == 'conflict': flags.append('dom_wide_grants=1')
    cmd = ['python3', str(a.runner.resolve()), '--qemu', str(a.qemu.resolve()),
           '--images', str(a.images.resolve()), '--firmware', str(a.firmware.resolve()),
           '--stage', str(stage.resolve()), '--work', str((work / 'vm').resolve()),
           '--memory', '8G', '--disk-mib', '64', '--timeout', '240', '--icount', '0',
           '--kernel-arg', ' '.join('capstone_vm.' + flag for flag in flags)]
    cpu = ','.join(['x-capstone-supervisor-slots=1', 'x-pcrel=false',
                    'x-capstone-raw-moves=true', 'x-capstone-test-csrs=false',
                    'x-capstone-checked-copy=true', 'x-capstone-context-transport=true'])
    env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA=cpu)
    (work / 'command.json').write_text(json.dumps({'argv': cmd, 'cpu_extra': cpu}, indent=2) + '\n')
    (a.out / 'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    with (work / 'runner.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    serial = (work / 'vm/serial.log').read_text(errors='replace')
    assert f'LOCAL_TMP_GATE_EXIT:{mode}:0\n' in serial
    assert not re.search(r'WARNING:|BUG:|Oops|Kernel panic|_FAIL|FAIL line', serial)
    for arm in ['exchange', 'direct']:
        body = serial.split(f'LOCAL_TMP_BEGIN:{mode}:{arm}\n')[1].split(f'LOCAL_TMP_END:{mode}:{arm}\n')[0]
        driver = re.search(rf'NATIVE_LOCAL_TMP_DRIVER mode={mode} checks=(\d+) failures=0\n', body)
        assert driver and int(driver[1]) > 0
        row = {'mode': mode, 'sdk': arm, 'driver_checks': int(driver[1]), 'failures': 0}
        if mode != 'conflict':
            native = re.search(rf'NATIVE_LOCAL_TMP mode={mode} checks=(\d+) failures=0\n', body)
            assert native and int(native[1]) > 0
            row['native_checks'] = int(native[1])
        if mode in ['on', 'on-null']:
            child = re.search(r'NATIVE_LOCAL_TMP_CHILD checks=(\d+) failures=0\n', body)
            assert child and int(child[1]) > 0
            row['child_checks'] = int(child[1])
        rows.append(row)
    print(json.dumps({'mode': mode, 'status': 'PASS', 'rows': rows[-2:]}), flush=True)
result = {'status': 'PASS', 'inputs': inputs, 'rows': rows, 'limitations': [
    'Single-hart root-only prototype; alias is an opt-in resource mapping, not chroot.',
    'Move-out tests exercise subsequent lookup refusal, not atomic revocation during racing renames.',
    'Previously admitted file descriptors retain their object authority.',
    'This fixture does not establish a passing SPEC workload.']}
(a.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
