#!/usr/bin/env python3
"""Fresh-guest GCC comparison: fixed task kernel, musl routing off/on."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
for name in ['qemu', 'images', 'firmware', 'plugin', 'profrun', 'gcc', 'gcc-direct', 'input', 'reference', 'out']:
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--routes', action='store_true', help='two diagnostic guests with route tracing; no instruction comparison')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
virtual = Path(__file__).resolve().parents[2]
inputs = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
          [('qemu', a.qemu), ('Image', a.images/'Image'), ('System.map', a.images/'System.map'),
           ('rootfs', a.images/'rootfs.ext2'), ('firmware', a.firmware), ('plugin', a.plugin),
           ('gcc', a.gcc), ('gcc_direct', a.gcc_direct), ('input', a.input), ('reference', a.reference), ('profrun', a.profrun)]}
(a.out/'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
syms = {v[2]: int(v[0], 16) for line in (a.images/'System.map').read_text().splitlines()
        if len(v := line.split()) >= 3}
rows = []
for index, direct in enumerate([0, 1] if a.routes else [0, 1, 1, 0]):
    d = a.out/f'{index}-direct-{direct}'
    stage = d/'stage'
    stage.mkdir(parents=True)
    (d/'profile').mkdir()
    for src, name in [(a.gcc_direct if direct else a.gcc, 'gcc.dom'), (a.input, 't1.c'), (a.reference, 'reference.s'), (a.profrun, 'profrun')]:
        shutil.copy2(src, stage/name)
    (stage/'gate.sh').write_text('''#!/bin/sh
set -eu
trap 'echo GCC_TASK_EXIT:$?; echo VIRTUAL_STAGED_DONE' EXIT
cp -a /mnt/vm /tmp/gcc-task
cd /tmp/gcc-task
[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status
echo 0 > /proc/sys/kernel/randomize_va_space
killall syslogd klogd 2>/dev/null || true
chmod +x profrun gcc.dom
./profrun ./gcc.dom t1.c -O3 -finline-limit=50000 -o output.s
cmp output.s reference.s
echo PASS:GCC_EXACT_OUTPUT
dmesg
''')
    flags = f'capstone_vm.native_task=1 capstone_vm.native_io=1 capstone_vm.native_route_trace={int(a.routes)} capstone_vm.dom_binfmt=1 capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1 capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1 capstone_vm.supervisor_slots=32'
    cmd = ['python3', str(virtual/'run-staged.py'), '--qemu', str(a.qemu.resolve()),
           '--images', str(a.images.resolve()), '--firmware', str(a.firmware.resolve()),
           '--stage', str(stage.resolve()), '--work', str((d/'vm').resolve()), '--disk-mib', '128',
           '--memory', '4G', '--timeout', '360', '--icount', '0', '--kernel-arg', flags,
           '--plugin', str(a.plugin.resolve()), '--plugin-arg', 'out='+str((d/'profile').resolve())]
    (d/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA='x-capstone-supervisor-slots=32,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true')
    with (d/'runner.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (d/'vm/serial.log').read_text(errors='replace')
    assert 'GCC_TASK_EXIT:0\n' in log and 'PASS:GCC_EXACT_OUTPUT\n' in log
    assert not re.search(r'WARNING:|BUG:|Oops|Kernel panic', log)
    if a.routes:
        # Ignore the final dmesg replay. Tracing is never used for timing.
        body = log.split('PASS:GCC_EXACT_OUTPUT\n', 1)[0]
        calls = [dict(route=route, nr=int(nr), count=int(count), result=int(result))
                 for route, nr, count, result in re.findall(
                     r'native_route route=(\w+) nr=(\d+) count=(\d+) result=(-?\d+)', body)]
        assert calls
        rows.append(dict(index=index, direct_io=bool(direct), calls=calls, reference_match=True))
        print(json.dumps(rows[-1]), flush=True)
        continue
    windows = list((d/'profile').glob('window-*.tsv'))
    assert len(windows) == 1 and not list((d/'profile').glob('*.invalid'))
    counts = dict(kernel=0, other=0)
    for line in windows[0].read_text().splitlines():
        if line.startswith('#'): continue
        fields = line.split(); pc, attempts = int(fields[0], 16), int(fields[3])
        counts['kernel' if syms['_stext'] <= pc < syms['_etext'] else 'other'] += attempts
    row = dict(index=index, direct_io=bool(direct), total_attempts=sum(counts.values()), classes=counts, reference_match=True)
    rows.append(row)
    print(json.dumps(row), flush=True)
(a.out/'summary.json').write_text(json.dumps(dict(inputs=inputs, diagnostic_routes=a.routes, rows=rows,
    limitations=['route counts are transport arguments, not generally original bytes; tracing is diagnostic only', 'attempted instructions, not wall time', 'same task kernel/config and program objects; musl routing off/on',
                 'built-in adapter counted as kernel; other includes firmware and userspace',
                 'two cold runs per libc variant; scoped prototype, not complete Linux compatibility']), indent=2)+'\n')
