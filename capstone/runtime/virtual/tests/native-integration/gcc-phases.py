#!/usr/bin/env python3
"""Partition a full SPEC GCC t1 run at authenticated instruction sites.
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
p.add_argument('--plain', type=Path, required=True)
p.add_argument('--repetitions', type=int, default=2)
p.add_argument('--arms', nargs='+', choices=['plain', 'launcher', 'native'], default=['plain', 'launcher', 'native'])
p.add_argument('--negative-boundary', action='store_true', help='Corrupt start-site fingerprint; validation must reject the run')
a = p.parse_args()
if a.repetitions < 1: p.error('positive repetitions required')
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
orders = [(f'{rep}-{arm}', [arm]) for rep in range(a.repetitions)
          for arm in (a.arms if rep % 2 == 0 else list(reversed(a.arms)))]
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
    arm = arms[0]
    exe = a.plain if arm == 'plain' else a.gcc
    resolver = virtual.parents[1]/'ports/spec-cpu2017/tools/stop-at.py'
    sites = json.loads(subprocess.check_output(['python3', str(resolver), 'resolve', '--nm', 'nm',
        '--kind', 'plain' if arm == 'plain' else 'capstone', '--exe', str(exe),
        '--symbol', 'statistics_fini', '--count', '1', '--start-symbol', 'init_cgraph', '--start-count', '1'], text=True))
    if a.negative_boundary:
        sites['plugin_args'] = [('start_bytes='+format(int(v.split('=')[1],16)^0x10,'x'))
                                if v.startswith('start_bytes=') else v for v in sites['plugin_args']]
    (d/'sites.json').write_text(json.dumps(sites, indent=2)+'\n')
    for arg in ['phases=1']+sites['plugin_args']: cmd += ['--plugin-arg', arg]
    (d/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    with (d/'runner.log').open('w') as log: subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (d/'vm/serial.log').read_text(errors='replace').replace('\r', '')
    assert 'GCC_TEST_EXIT:0\n' in log
    assert not re.search(r'WARNING:|BUG:|Oops|Kernel panic', log)
    assert len(list((d/'capprof').glob('window-*.tsv'))) == 3, 'missing phase boundary'
    assert not list((d/'capprof').glob('*.invalid')), 'invalid phase boundary'
    assert (d/'output/rc-0.txt').read_text().strip() == '0'
    assert (d/'output/output-0.s').read_bytes() == a.reference.read_bytes()
    block = log.split(f'GCC_BEGIN:0:{arm}\n', 1)[1].split('GCC_END:0\n', 1)[0]
    bases = {n: int(v,16) for n,v in re.findall(r'^SECTION (\S+) (0x[0-9a-f]+)$', block,re.M)}
    assert '.text' in bases
    ranges = [(bases[n], bases[n]+length) for n,length in sections if n in bases]
    totals = []
    for index, phase in enumerate(['startup_and_initialization', 'compilation_and_output', 'finalization_and_exit']):
        counts = dict(kernel=0,module=0,other=0)
        headers_seen = []
        with (d/f'capprof/window-{index}.tsv').open() as stream:
            for line in stream:
                if line.startswith('#'):
                    headers_seen.append(line.strip()); continue
                v=line.split(); pc,attempts=int(v[0],16),int(v[3])
                category='kernel' if kernel[0]<=pc<kernel[1] else 'module' if any(lo<=pc<hi for lo,hi in ranges) else 'other'
                counts[category]+=attempts
        header='\n'.join(headers_seen)
        total=sum(counts.values())
        assert total == int(re.search(r'^# total (\d+)',header,re.M)[1])
        if index < 2:
            label='started_at' if index==0 else 'stopped_at'
            assert re.search(r'^# '+label+r'=[0-9a-f]+:1 .*anchor_hits=[1-9][0-9]* broken=0 armed_bases=1$',header,re.M), header
            # Anchors count arrivals (fault retries included); boundary entries
            # use the site's successor and must identify one physical candidate.
            which='start' if index==0 else 'stop'
            candidates=re.search(r'^# '+which+r'_candidates (.*)$',header,re.M)[1].split()
            used=[v.split(':') for v in candidates if int(v.split(':')[2])]
            assert len(used)==1 and used[0][2:] == ['1','1','0'], candidates
        totals.append(total)
        rows.append(dict(run=order,arm=arm,phase=phase,total_attempts=total,classes=counts,reference_match=True))
        print(rows[-1],flush=True)
    outer=int((d/'capprof/phase-total.txt').read_text())
    assert sum(totals)==outer, (totals,outer)
    (d/'verdict.json').write_text(json.dumps(dict(pass_=True,outer_attempts=outer,phase_sum=sum(totals),reference_match=True),indent=2)+'\n')

means={arm:{phase:sum(r['total_attempts'] for r in rows if r['arm']==arm and r['phase']==phase)/a.repetitions
            for phase in ['startup_and_initialization','compilation_and_output','finalization_and_exit']} for arm in a.arms}
result=dict(inputs_sha256=inputs,rows=rows,means=means,repetitions=a.repetitions,
    boundaries=['after first init_cgraph instruction','after first statistics_fini instruction'],
    scope='One GCC invocation per fresh guest; all runs cold; full reference validation; phases sum to independently snapshotted outer total',
    limitations=['compilation includes compile_file initialization and output-file close; startup includes GCC backend/language initialization',
                 'same cached binaries and native syscall limitations as three-arm comparison',
                 'cold totals are not directly comparable with mixed warm/cold three-arm means',
                 'QEMU attempts, not hardware timing'])
(a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS phase boundaries, conservation, exact output')
