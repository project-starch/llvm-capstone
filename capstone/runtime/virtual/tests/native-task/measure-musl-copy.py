#!/usr/bin/env python3
"""Matched musl file-copy workload, setup excluded, fresh guests in ABBA order."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess

p = argparse.ArgumentParser(description=__doc__)
for name in ['qemu', 'images', 'firmware', 'plugin', 'probes', 'out']:
    p.add_argument('--'+name, type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
virtual = Path(__file__).resolve().parents[2]
inputs = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
          [('qemu', a.qemu), ('Image', a.images/'Image'), ('System.map', a.images/'System.map'),
           ('rootfs', a.images/'rootfs.ext2'), ('firmware', a.firmware), ('plugin', a.plugin)] +
          [(name, a.probes/name) for name in ['musl-copy.o', 'copy-exchange.dom', 'copy-direct.dom']]}
(a.out/'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
syms = {v[2]: int(v[0], 16) for line in (a.images/'System.map').read_text().splitlines() if len(v := line.split()) >= 3}
rows = []
for run, arm in enumerate(['exchange', 'direct', 'direct', 'exchange']):
    d = a.out/f'{run}-{arm}'
    stage = d/'stage'; stage.mkdir(parents=True)
    (d/'profile').mkdir()
    shutil.copy2(a.probes/f'copy-{arm}.dom', stage/'copy.dom')
    cases = [64, 4096, 65536] if run < 2 else [65536, 4096, 64]
    lines = ['#!/bin/sh', 'set -eu', "trap 'echo COPY_EXIT:$?; dmesg; echo VIRTUAL_STAGED_DONE' EXIT",
             'cp -a /mnt/vm /tmp/musl-copy', 'cd /tmp/musl-copy', 'chmod +x copy.dom',
             '[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status',
             'echo 0 > /proc/sys/kernel/randomize_va_space', 'killall syslogd klogd 2>/dev/null || true']
    for size in cases:
        lines += [f'echo CASE:{size}', f'./copy.dom {size}', f'echo END:{size}']
    (stage/'gate.sh').write_text('\n'.join(lines)+'\n')
    flags = 'capstone_vm.native_task=1 capstone_vm.native_io=1 capstone_vm.dom_binfmt=1 capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1 capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1 capstone_vm.supervisor_slots=32'
    cmd = ['python3', str(virtual/'run-staged.py'), '--qemu', str(a.qemu.resolve()),
           '--images', str(a.images.resolve()), '--firmware', str(a.firmware.resolve()),
           '--stage', str(stage.resolve()), '--work', str((d/'vm').resolve()), '--disk-mib', '64',
           '--memory', '4G', '--timeout', '360', '--icount', '0', '--kernel-arg', flags,
           '--plugin', str(a.plugin.resolve()), '--plugin-arg', 'out='+str((d/'profile').resolve())]
    (d/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA='x-capstone-supervisor-slots=32,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true')
    with (d/'runner.log').open('w') as out:
        subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
    log = (d/'vm/serial.log').read_text(errors='replace')
    assert 'COPY_EXIT:0\n' in log and not re.search(r'WARNING:|BUG:|Oops|Kernel panic', log)
    assert len(list((d/'profile').glob('window-*.tsv'))) == len(cases)
    assert not list((d/'profile').glob('*.invalid'))
    for index, size in enumerate(cases):
        calls = 2 * (1048576 // size)
        body = log.split(f'CASE:{size}\n', 1)[1].split(f'END:{size}\n', 1)[0]
        assert f'MUSL_COPY bytes=1048576 chunk={size} calls={calls} failures=0\n' in body
        counts = dict(kernel=0, other=0)
        for line in (d/'profile'/f'window-{index}.tsv').read_text().splitlines():
            if line.startswith('#'): continue
            fields = line.split(); pc, attempts = int(fields[0], 16), int(fields[3])
            counts['kernel' if syms['_stext'] <= pc < syms['_etext'] else 'other'] += attempts
        rows.append(dict(run=run, arm=arm, chunk=size, calls=calls, bytes=1048576,
                         total_attempts=sum(counts.values()), classes=counts, output_verified=True))
    print('PASS', run, arm, flush=True)
means = []
for size in [64, 4096, 65536]:
    arms = {}
    for arm in ['exchange', 'direct']:
        rs = [r for r in rows if r['arm'] == arm and r['chunk'] == size]
        arms[arm] = dict(total_attempts=statistics.mean(r['total_attempts'] for r in rs),
                        classes={k: statistics.mean(r['classes'][k] for r in rs) for k in ['kernel', 'other']})
    baseline, direct = arms['exchange'], arms['direct']
    means.append(dict(chunk=size, calls=2*(1048576//size), arms=arms,
                      total_reduction_percent=100*(1-direct['total_attempts']/baseline['total_attempts']),
                      kernel_reduction_percent=100*(1-direct['classes']['kernel']/baseline['classes']['kernel'])))
result = dict(inputs=inputs, rows=rows, means=means, limitations=[
    'System-wide attempted instructions during read/write loop; not cycles or wall-clock speed.',
    'Real fixed-prototype musl read/write; same application object, two SDK variants, same task kernel.',
    'Resident regular files, fixed 1 MiB copied; setup, verification and diagnostic output excluded.',
    'Two repeats per arm/case; ABBA guest order and reversed case order; checked-copy rules unchanged.',
    'Loop and return-count checks included. Kernel includes built-in adapter; other includes userspace/firmware.'])
(a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
for row in means: print(json.dumps(row), flush=True)
