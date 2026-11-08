#!/usr/bin/env python3
"""Qualify opt-in musl IO on the built-in task kernel; source test env first."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

here = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('mode', choices=['io', 'race', 'race-file'])
for name in ['qemu', 'images', 'firmware', 'probes', 'out']:
    p.add_argument('--'+name, type=Path, required=True)
p.add_argument('--feature', type=int, choices=[0, 1], default=1)
p.add_argument('--fast', type=int, choices=[0, 1], default=1)
p.add_argument('--locks', type=int, choices=[0, 1], default=0)
p.add_argument('--stat', type=int, choices=[0, 1], default=0)
p.add_argument('--seek', type=int, choices=[0, 1], default=0)
p.add_argument('--fd', type=int, choices=[0, 1], default=0)
p.add_argument('--close', type=int, choices=[0, 1], default=0)
p.add_argument('--path', type=int, choices=[0, 1], default=0)
p.add_argument('--clock', type=int, choices=[0, 1], default=0)
p.add_argument('--kernel-arg', action='append', default=[])
p.add_argument('--no-icount', action='store_true', help='Use wall-clock virtual time for functional tests; no deterministic instruction counts')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
stage = a.out/'stage'
stage.mkdir()
inputs = {}
def copy(src, name=None):
    name = name or src.name
    shutil.copy2(src, stage/name)
    inputs[name] = hashlib.sha256(src.read_bytes()).hexdigest()
for name, src in [('qemu', a.qemu), ('Image', a.images/'Image'), ('firmware', a.firmware), ('rootfs', a.images/'rootfs.ext2')]:
    inputs[name] = hashlib.sha256(src.read_bytes()).hexdigest()
lines = ['#!/bin/sh', 'set -eu', "trap 'echo MUSL_GATE_EXIT:$?; dmesg; echo VIRTUAL_STAGED_DONE' EXIT",
         'cp -a /mnt/vm /tmp/musl-gate', 'cd /tmp/musl-gate', 'chmod +x *.dom',
         '[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status',
         'echo 0 > /proc/sys/kernel/randomize_va_space']
scenarios = []
if a.mode == 'io':
    for name in ['io-exchange.dom', 'io-direct.dom', 'musl-pipe', 'probe.dom', 'blocked.dom', 'entry-driver', 'gcc-coverage.dom', 'compute.dom', 'task-driver']:
        copy(a.probes/name)
    lines += ['chmod +x musl-pipe entry-driver task-driver']
    for arm in ['exchange', 'direct']:
        lines += [f'echo MUSL_BEGIN:{arm}', f'./io-{arm}.dom', f'./musl-pipe ./io-{arm}.dom', f'echo MUSL_END:{arm}']
    lines += ['echo 0 > /sys/module/capstone_vm/parameters/native_route_trace',
              'yes Q | head -c 131072 > input', './probe.dom <input >output', 'cat output',
              './gcc-coverage.dom', './entry-driver ./blocked.dom', './task-driver ./compute.dom ./blocked.dom']
else:
    for name in ['race.dom', 'pipe-fill']: copy(a.probes/name)
    lines += ['chmod +x pipe-fill', r"head -c 131072 /dev/zero | tr '\000' Q > input"]
    source = here.parent/'native-race/scenarios.txt'
    shutil.copy2(source, a.out/'scenarios.txt')
    for row in source.read_text().splitlines():
        if not row or row.startswith('#'): continue
        name, params, args = row.split('|')
        scenarios.append(name)
        for option in ['rt_norecheck', 'rt_noexcl', 'rt_dummy', 'rt_nohold']:
            lines += [f'echo {int(option in params.split())} > /sys/module/capstone_vm/parameters/{option}']
        command = './pipe-fill ./race.dom' if a.mode == 'race' else './race.dom'
        command += ' ' + ' '.join(shlex.quote(x) for x in args.split())
        if a.mode == 'race-file': command += ' <input'
        lines += [f'echo NATIVE_CASE:{name}', 'status=0', command+' || status=$?', 'echo RC=$status', f'echo NATIVE_END:{name}']
if any('capstone_vm.unpinned_access_fault=' in arg for arg in a.kernel_arg):
    lines.insert(4, 'echo ACCESS_POLICY:$(cat /sys/module/capstone_vm/parameters/unpinned_access_fault)')
(stage/'gate.sh').write_text('\n'.join(lines)+'\n')
(a.out/'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
slots = 1 if a.mode == 'io' else 32
flags = f'capstone_vm.native_task=1 capstone_vm.native_io={a.feature} capstone_vm.native_locks={a.locks} capstone_vm.native_stat={a.stat} capstone_vm.native_seek={a.seek} capstone_vm.native_fd={a.fd} capstone_vm.native_close={a.close} capstone_vm.native_path={a.path} capstone_vm.native_clock={a.clock} capstone_vm.native_fast={a.fast} capstone_vm.native_route_trace={int(a.mode == "io")} capstone_vm.dom_binfmt=1 capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1 capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1 capstone_vm.supervisor_slots={slots}'
# These features default off; omit redundant zeros to fit the kernel cmdline.
flags = ' '.join(item for item in flags.split() if item not in {f'capstone_vm.native_{key}=0' for key in ['io','locks','stat','seek','fd','close','path','clock','route_trace']})
assert len(('root=/dev/vda ro '+flags+' '+' '.join(a.kernel_arg)).encode()) < 512, 'kernel command line would be truncated'
cmd = ['python3', str(here.parents[1]/'run-staged.py'), '--qemu', str(a.qemu.resolve()), '--images', str(a.images.resolve()), '--firmware', str(a.firmware.resolve()), '--stage', str(stage.resolve()), '--work', str((a.out/'vm').resolve()), '--disk-mib', '64', '--memory', '4G', '--timeout', '360', '--kernel-arg', flags]
if not a.no_icount: cmd += ['--icount', '0']
for arg in a.kernel_arg: cmd += ['--kernel-arg', arg]
(a.out/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA=f'x-capstone-supervisor-slots={slots},x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true')
with (a.out/'runner.log').open('w') as log:
    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
log = (a.out/'vm/serial.log').read_text(errors='replace').replace('\r', '')
for arg in a.kernel_arg:
    for item in arg.split():
        if item.startswith('capstone_vm.unpinned_access_fault='):
            assert 'ACCESS_POLICY:'+('Y' if item.endswith('=1') else 'N')+'\n' in log
assert 'MUSL_GATE_EXIT:0\n' in log, log[-6000:]
assert not re.search(r'WARNING:|BUG:|Oops|Kernel panic', log)
if a.mode == 'io':
    assert 'native integration checks=51 failures=0\n' in log
    assert 'native entry failures=0\n' in log and 'native task failures=0\n' in log
    for arm in ['exchange', 'direct']:
        body = log.split('MUSL_BEGIN:'+arm+'\n', 1)[1].split('MUSL_END:'+arm+'\n', 1)[0]
        results = re.findall(r'MUSL_IO checks=(\d+) failures=(\d+)', body)
        expected = 176 if arm == 'direct' and a.feature else 170
        assert results == [(str(expected), '0'), ('4', '0')], (arm, results)
        assert not re.search(r'native_route .* nr=172 ', body), 'cached getpid escaped'
        spans = re.findall(r'TRANSFER_BEGIN:(\d+)\n(.*?)TRANSFER_END:\1\n', body, re.S)
        assert len(spans) == 10, len(spans)
        for size, span in spans:
            routes = re.findall(r'native_route route=(\w+) nr=(63|64) count=(\d+) result=(-?\d+)', span)
            direct = arm == 'direct' and a.feature and 0 < int(size) <= 65536
            assert routes and all(r[0] == ('register' if direct else 'exchange') for r in routes), (arm, size, routes)
            if direct: assert len(routes) == 2 and all(r[2] == size and r[3] == size for r in routes)
    (a.out/'verdict.txt').write_text('PASS musl IO, scalar shape, route boundaries, cached identity, authority, entry and task lifecycle\n')
else:
    for name in scenarios:
        body = log.split('NATIVE_CASE:'+name+'\n', 1)[1].split('NATIVE_END:'+name+'\n', 1)[0]
        assert 'RC=0\n' in body, name
        (a.out/(name+'.log')).write_text(body)
    with (a.out/'verdict.txt').open('w') as out:
        subprocess.run(['python3', str(here.parent/'native-race/judge.py'), str(a.out)], stdout=out, check=True)
print('PASS', a.mode, a.out)
