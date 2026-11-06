#!/usr/bin/env python3
"""SQLite speedtest1 file-backed comparison with 32 host-reference oracles."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
for name in ['qemu', 'images', 'firmware', 'plugin', 'profrun', 'build', 'reference', 'out']:
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--routes', action='store_true', help='two diagnostic guests with route tracing; no instruction comparison')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
virtual = Path(__file__).resolve().parents[2]
inputs = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
          [('qemu', a.qemu), ('Image', a.images/'Image'), ('System.map', a.images/'System.map'),
           ('rootfs', a.images/'rootfs.ext2'), ('firmware', a.firmware), ('plugin', a.plugin),
           ('exchange', a.build/'speedtest1-exchange.dom'), ('direct', a.build/'speedtest1-direct.dom'), ('sqlite3.o', a.build/'sqlite3.o'), ('speedtest1.o', a.build/'speedtest1.o'), ('reference', a.reference), ('profrun', a.profrun)]}
(a.out/'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
syms = {v[2]: int(v[0], 16) for line in (a.images/'System.map').read_text().splitlines()
        if len(v := line.split()) >= 3}
reference = [line for line in a.reference.read_text().splitlines() if line.startswith('STUDY-ORACLE ')]
assert len(reference) == 32
rows = []
for index, direct in enumerate([0, 1] if a.routes else [0, 1, 1, 0]):
    d = a.out/f'{index}-direct-{direct}'
    stage = d/'stage'
    stage.mkdir(parents=True)
    (d/'profile').mkdir()
    for src, name in [(a.build/('speedtest1-direct.dom' if direct else 'speedtest1-exchange.dom'), 'sqlite.dom'), (a.profrun, 'profrun')]:
        shutil.copy2(src, stage/name)
    (stage/'gate.sh').write_text('''#!/bin/sh
set -eu
trap 'echo SQLITE_TASK_EXIT:$?; echo VIRTUAL_STAGED_DONE' EXIT
cp -a /mnt/vm /tmp/sqlite-task
cd /tmp/sqlite-task
[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status
echo 0 > /proc/sys/kernel/randomize_va_space
killall syslogd klogd 2>/dev/null || true
chmod +x profrun sqlite.dom
./profrun ./sqlite.dom --size 1 st.db >sqlite.out 2>sqlite.err
cat sqlite.out sqlite.err
echo SQLITE_OUTPUT_END
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
    assert 'SQLITE_TASK_EXIT:0\n' in log and 'SQLITE_OUTPUT_END\n' in log
    oracle = [line for line in log.splitlines() if line.startswith('STUDY-ORACLE ')]
    assert oracle == reference
    assert not re.search(r'WARNING:|BUG:|Oops|Kernel panic', log)
    if a.routes:
        # Ignore the final dmesg replay. Tracing is never used for timing.
        body = log.split('SQLITE_OUTPUT_END\n', 1)[0]
        calls = [dict(route=route, nr=int(nr), count=int(count), result=int(result))
                 for route, nr, count, result in re.findall(
                     r'native_route route=(\w+) nr=(\d+) count=(\d+) result=(-?\d+)', body)]
        assert calls
        rows.append(dict(index=index, direct_io=bool(direct), calls=calls, reference_match=True, oracle_phases=32))
        print(json.dumps(rows[-1]), flush=True)
        continue
    windows = list((d/'profile').glob('window-*.tsv'))
    assert len(windows) == 1 and not list((d/'profile').glob('*.invalid'))
    counts = dict(kernel=0, other=0)
    for line in windows[0].read_text().splitlines():
        if line.startswith('#'): continue
        fields = line.split(); pc, attempts = int(fields[0], 16), int(fields[3])
        counts['kernel' if syms['_stext'] <= pc < syms['_etext'] else 'other'] += attempts
    row = dict(index=index, direct_io=bool(direct), total_attempts=sum(counts.values()), classes=counts, reference_match=True, oracle_phases=32)
    rows.append(row)
    print(json.dumps(row), flush=True)
(a.out/'summary.json').write_text(json.dumps(dict(inputs=inputs, diagnostic_routes=a.routes, rows=rows,
    limitations=['route counts are transport arguments, not generally original bytes; tracing is diagnostic only', 'attempted instructions, not wall time', 'same task kernel/config and program objects; musl routing off/on',
                 'built-in adapter counted as kernel; other includes firmware and userspace',
                 'two cold runs per libc variant; scoped prototype, not complete Linux compatibility', '32 stock host SQLite SQL phase oracles; default journaling/sync/locking enabled', 'full process window including startup, file IO and oracle calculations; tmpfs database, no physical storage durability claim']), indent=2)+'\n')
