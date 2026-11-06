#!/usr/bin/env python3
"""Matched SPEC GCC t1 test, existing exchange ABI; no benchmark sources vendored.
Source capstone/tests/capstone-test-env.sh before invoking. All input paths are
explicit; --out must be fresh. Requires matching Image/rootfs.ext2/System.map.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

p = argparse.ArgumentParser(description=__doc__)
for n in ['qemu', 'images', 'firmware', 'module', 'launcher', 'gcc', 'profrun', 'input', 'reference', 'plugin', 'out']:
    p.add_argument('--'+n, type=Path, required=True)
p.add_argument('--plain', type=Path, help='Optional ordinary RV64 GCC; three position-balanced guest runs')
p.add_argument('--reverse-rotations', action='store_true', help='With --plain, use the other three orders; combine both batches for all six permutations')
a = p.parse_args()
if a.reverse_rotations and not a.plain: p.error('--reverse-rotations requires --plain')
a.out.mkdir(parents=True, exist_ok=False)
virtual = Path(__file__).resolve().parents[2]
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
inputs = {n: sha(getattr(a, n)) for n in ['qemu', 'firmware', 'module', 'launcher', 'gcc', 'profrun', 'input', 'reference', 'plugin']}
if a.plain: inputs['plain'] = sha(a.plain)
inputs.update({n: sha(a.images/n) for n in ['Image', 'System.map']})
syms = {v[2]: int(v[0], 16) for line in (a.images/'System.map').read_text().splitlines() if len(v := line.split()) >= 3}
kernel = syms['_stext'], syms['_etext']
data = a.module.read_bytes()
assert data[:6] == b'\x7fELF\x02\x01'
off = struct.unpack_from('<Q', data, 0x28)[0]
size, count, names = struct.unpack_from('<HHH', data, 0x3a)
headers = [struct.unpack_from('<IIQQQQIIQQ', data, off+i*size) for i in range(count)]
strings = headers[names][4]
sections = [(data[strings+h[0]:data.index(b'\0', strings+h[0])].decode(), h[5]) for h in headers if h[2] & 4]
rows = []
orders = [('forward', ['launcher', 'native']), ('reverse', ['native', 'launcher'])]
if a.plain:
    orders = [('rotation-0', ['plain', 'launcher', 'native']),
              ('rotation-1', ['launcher', 'native', 'plain']),
              ('rotation-2', ['native', 'plain', 'launcher'])]
if a.reverse_rotations:
    orders = [('reverse-'+name, list(reversed(arms))) for name, arms in orders]
for order, arms in orders:
    d = a.out/order
    stage = d/'stage'
    stage.mkdir(parents=True)
    (d/'output').mkdir()
    (d/'capprof').mkdir()
    for field, name in [('module', 'capstone_vm.ko'), ('launcher', 'capstone-vexec'), ('gcc', 'gcc-test.dom'), ('profrun', 'profrun'), ('input', 't1.c')]:
        shutil.copy2(getattr(a, field), stage/name)
    if a.plain: shutil.copy2(a.plain, stage/'gcc-plain')
    lines = ['#!/bin/sh', 'set -eu', "trap 'echo GCC_TEST_EXIT:$?; echo VIRTUAL_STAGED_DONE' EXIT",
             '[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status',
             'mkdir -p /mnt/results', 'mount -t 9p -o trans=virtio,version=9p2000.L gate9p /mnt/results',
             'cp -a /mnt/vm /tmp/gcc-test', 'cd /tmp/gcc-test', 'dmesg -n 1',
             'killall syslogd klogd 2>/dev/null || true', 'echo 0 > /proc/sys/kernel/randomize_va_space']
    for index, arm in enumerate(arms):
        flags = 'supervisor_slots=32 unpinned=1'
        if arm == 'native': flags += ' dom_binfmt=1 native_dispatch=1 checked_copy=1 native_slot_transport=1 native_fast=1'
        command = {'native': './gcc-test.dom', 'launcher': './capstone-vexec gcc-test.dom', 'plain': './gcc-plain'}[arm]
        lines += [f'echo GCC_BEGIN:{index}:{arm}', f'insmod capstone_vm.ko {flags}',
                  'for f in /sys/module/capstone_vm/sections/.*; do [ ! -f "$f" ] || echo SECTION ${f##*/} $(cat "$f"); done',
                  'rm -f output.s', 'rc=0',
                  f'./profrun {command} t1.c -O3 -finline-limit=50000 -o output.s >stdout.txt 2>stderr.txt || rc=$?',
                  'echo GCC_RC:$rc', 'cat stderr.txt', 'echo REFUSED:$(cat /sys/module/capstone_vm/parameters/native_refused_calls)',
                  f'echo "$rc" > /mnt/results/rc-{index}.txt', f'cp stdout.txt /mnt/results/stdout-{index}.txt', f'cp stderr.txt /mnt/results/stderr-{index}.txt',
                  f'[ ! -f output.s ] || cp output.s /mnt/results/output-{index}.s',
                  'rmmod capstone_vm', f'echo GCC_END:{index}']
    lines += ['dmesg']
    (stage/'gate.sh').write_text('\n'.join(lines)+'\n')
    env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA='x-capstone-supervisor-slots=32,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true')
    cmd = ['python3', str(virtual/'run-staged.py'), '--qemu', str(a.qemu.resolve()), '--images', str(a.images.resolve()),
           '--firmware', str(a.firmware.resolve()), '--stage', str(stage.resolve()), '--work', str((d/'vm').resolve()),
           '--share', str((d/'output').resolve()), '--plugin', str(a.plugin.resolve()), '--plugin-arg', 'out='+str((d/'capprof').resolve()),
           '--icount', '0', '--memory', '4G', '--disk-mib', '128', '--timeout', '360']
    (d/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    with (d/'runner.log').open('w') as log: subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (d/'vm/serial.log').read_text(errors='replace').replace('\r', '')
    assert 'GCC_TEST_EXIT:0\n' in log
    assert not re.search(r'WARNING:|BUG:|Oops|Kernel panic', log)
    assert len(list((d/'capprof').glob('window-*.tsv'))) == len(arms) and not list((d/'capprof').glob('*.invalid'))
    for index, arm in enumerate(arms):
        block = log.split(f'GCC_BEGIN:{index}:{arm}\n', 1)[1].split(f'GCC_END:{index}\n', 1)[0]
        assert (d/f'output/rc-{index}.txt').read_text().strip() == '0'
        assert (d/f'output/output-{index}.s').read_bytes() == a.reference.read_bytes()
        bases = {n: int(v, 16) for n, v in re.findall(r'^SECTION (\S+) (0x[0-9a-f]+)$', block, re.M)}
        assert '.text' in bases
        ranges = [(bases[n], bases[n]+length) for n, length in sections if n in bases]
        counts = dict(kernel=0, module=0, other=0)
        for line in (d/f'capprof/window-{index}.tsv').read_text().splitlines():
            if line.startswith('#'): continue
            v = line.split(); pc, attempts = int(v[0], 16), int(v[3])
            category = 'kernel' if kernel[0] <= pc < kernel[1] else 'module' if any(lo <= pc < hi for lo, hi in ranges) else 'other'
            counts[category] += attempts
        rows.append(dict(order=order, arm=arm, total_attempts=sum(counts.values()), classes=counts, reference_match=True,
                         counted_refusals=int(re.search(r'^REFUSED:(\d+)$', block, re.M)[1])))
        print(rows[-1], flush=True)
means = {arm: sum(r['total_attempts'] for r in rows if r['arm'] == arm)/sum(r['arm'] == arm for r in rows) for arm in sorted({r['arm'] for r in rows})}
result = dict(benchmark='502.gcc_r', input='test/t1.c', argv=['t1.c', '-O3', '-finline-limit=50000', '-o', 'output.s'],
              inputs_sha256=inputs, rows=rows, means=means, reduction_percent=100*(1-means['native']/means['launcher']),
              scope='QEMU attempted instructions, profrun fork through child exit; fresh guests; reversed two-arm order or three cyclic positions; boot/module load outside windows',
              limitations=['existing exchange ABI and ioctl entry, not final native task integration', 'unsignalled test; caught SIGACTION and some ioctl requests still refused',
                           'refusal counter is not a complete syscall inventory', 'other PCs include application, runtime, launcher and firmware',
                           'two repetitions per arm (three with --plain), preliminary only; no hardware timing claim'])
(a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
if a.plain:
    result['ratios_to_plain'] = {arm: value/means['plain'] for arm, value in means.items()}
    result['limitations'] += ['plain is an ordinary RV64 application on the same patched Linux/model, not a stock-kernel comparison', 'existing plain build uses its recorded toolchain/libc; not a compiler-revision-matched isolation of ISA overhead']
    (a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
print('PASS exact reference in', len(rows), 'runs; reduction_percent=', result['reduction_percent'])
