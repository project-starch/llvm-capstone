#!/usr/bin/env python3
"""Factorial entry/argument-ABI microbenchmarks using the existing budget fixture."""
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
p.add_argument('--fast', type=int, choices=[0, 1], default=1)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
virtual = Path(__file__).resolve().parents[2]
binaries = ['budget-plain', 'budget-exchange.dom', 'budget-register.dom', 'budget-exchange-scalar.dom', 'pipe-exec']
inputs = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
          [('qemu', a.qemu), ('Image', a.images/'Image'), ('System.map', a.images/'System.map'),
           ('rootfs', a.images/'rootfs.ext2'), ('firmware', a.firmware), ('plugin', a.plugin)] +
          [(name, a.probes/name) for name in binaries]}
(a.out/'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
syms = {v[2]: int(v[0], 16) for line in (a.images/'System.map').read_text().splitlines()
        if len(v := line.split()) >= 3}
rows = []
cases = [('getpid', 0, 'read'), ('read1', 1, 'read'), ('read64K', 65536, 'read'),
         ('write1', 1, 'write'), ('write64K', 65536, 'write')]
for run, task in enumerate([0, 1, 1, 0]):
    d = a.out/f'{run}-task-{task}'
    stage = d/'stage'; stage.mkdir(parents=True)
    (d/'profile').mkdir()
    for name in binaries: shutil.copy2(a.probes/name, stage/name)
    arms = ['plain', 'exchange', 'exchange-scalar', 'register']
    if run >= 2: arms.reverse()
    lines = ['#!/bin/sh', 'set -eu', "trap 'echo SYSCALL_EXIT:$?; echo VIRTUAL_STAGED_DONE' EXIT",
             'cp -a /mnt/vm /tmp/syscalls', 'cd /tmp/syscalls', 'chmod +x budget-* pipe-exec',
             '[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status',
             'echo 0 > /proc/sys/kernel/randomize_va_space', 'killall syslogd klogd 2>/dev/null || true']
    windows = []
    for arm in arms:
        binary = 'budget-plain' if arm == 'plain' else f'budget-{arm}.dom'
        for name, size, direction in cases:
            if arm == 'exchange-scalar' and name != 'getpid': continue
            index = len(windows); windows.append((arm, name, size, direction))
            lines += [f'echo CASE:{index}:{arm}:{name}',
                      f'./pipe-exec ./{binary} {size} {direction} >budget.out',
                      'tail -n 1 budget.out']
            if direction == 'write':
                count = size if size == 65536 else 100
                lines += [f'test "$(head -c {count} budget.out | tr -d X | wc -c)" -eq 0']
            lines += [f'echo END:{index}']
    lines += ['dmesg']
    (stage/'gate.sh').write_text('\n'.join(lines)+'\n')
    flags = f'capstone_vm.native_fast={a.fast} capstone_vm.native_task={task} capstone_vm.dom_binfmt=1 capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1 capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1 capstone_vm.supervisor_slots=32'
    cmd = ['python3', str(virtual/'run-staged.py'), '--qemu', str(a.qemu.resolve()),
           '--images', str(a.images.resolve()), '--firmware', str(a.firmware.resolve()),
           '--stage', str(stage.resolve()), '--work', str((d/'vm').resolve()), '--disk-mib', '64',
           '--memory', '4G', '--timeout', '360', '--icount', '0', '--kernel-arg', flags,
           '--plugin', str(a.plugin.resolve()), '--plugin-arg', 'out='+str((d/'profile').resolve())]
    (d/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA='x-capstone-supervisor-slots=32,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true')
    with (d/'runner.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (d/'vm/serial.log').read_text(errors='replace')
    assert 'SYSCALL_EXIT:0\n' in log
    assert not re.search(r'WARNING:|BUG:|Oops|Kernel panic', log)
    assert len(list((d/'profile').glob('window-*.tsv'))) == len(windows)
    assert not list((d/'profile').glob('*.invalid'))
    for index, (arm, name, size, direction) in enumerate(windows):
        calls = 1 if size == 65536 else 100
        body = log.split(f'CASE:{index}:{arm}:{name}\n')[1].split(f'END:{index}\n')[0]
        assert f'BUDGET bytes={size} calls={calls} failures=0\n' in body
        counts = dict(kernel=0, other=0)
        for line in (d/'profile'/f'window-{index}.tsv').read_text().splitlines():
            if line.startswith('#'): continue
            fields = line.split(); pc, attempts = int(fields[0], 16), int(fields[3])
            counts['kernel' if syms['_stext'] <= pc < syms['_etext'] else 'other'] += attempts
        rows.append(dict(run=run, task=bool(task), abi=arm, case=name, calls=calls,
                         total_per_call=sum(counts.values())/calls,
                         kernel_per_call=counts['kernel']/calls, other_per_call=counts['other']/calls))
    print(f'run {run} task={task}: {len(windows)} windows passed', flush=True)
means = []
for task in [False, True]:
    for arm in ['plain', 'exchange', 'exchange-scalar', 'register']:
        for name, _, _ in cases:
            if arm == 'exchange-scalar' and name != 'getpid': continue
            group = [r for r in rows if (r['task'], r['abi'], r['case']) == (task, arm, name)]
            means.append(dict(task=task, abi=arm, case=name,
                **{key: statistics.mean(r[key] for r in group) for key in ['total_per_call', 'kernel_per_call', 'other_per_call']}))
result = dict(inputs=inputs, native_fast=bool(a.fast), rows=rows, means=means, limitations=[
    'system-wide attempted instructions, not cycles or wall time; includes loop and call wrappers',
    'two repeats per entry/ABI/case; four fresh guests, reverse ABI order in second pair',
    'getpid/1-byte: 100 calls; 64KiB: one call per window, prefaulted buffer and prefilled pipe',
    'read return lengths and write bytes checked; separate authority/copy suite validates semantics',
    'plain RV64 static Linux toolchain reference uses same patched kernel, not a stock-kernel comparison',
    'exchange getpid is cached, zero kernel entry; exchange-scalar forces real exchange getpid',
    'direct register probes exist; GCC still uses exchange ABI; not whole-libc migration'])
(a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
for row in means: print(json.dumps(row))
