#!/usr/bin/env python3
"""Matched kernels, fixed musl copy workload; run under run-quick-gate.py."""
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
for name in ['qemu', 'baseline', 'candidate', 'firmware', 'plugin', 'probes', 'out']:
    p.add_argument('--' + name, type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
virtual = Path(__file__).resolve().parents[2]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
inputs = {name: sha(getattr(a, name)) for name in ['qemu', 'firmware', 'plugin']}
for version in ['baseline', 'candidate']:
    images = getattr(a, version)
    inputs[version] = {name: sha(images/name) for name in ['Image', 'System.map', 'rootfs.ext2']}
assert inputs['baseline']['rootfs.ext2'] == inputs['candidate']['rootfs.ext2']
inputs['programs'] = {arm: sha(a.probes/('copy-' + arm + '.dom')) for arm in ['exchange', 'direct']}
(a.out/'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
rows = []
# Identical binaries and routes per comparison; fast=1 is the existing reuse control.
for fast in [0, 1]:
    for run, version in enumerate(['baseline', 'candidate', 'candidate', 'baseline']):
        images = getattr(a, version)
        syms = {v[2]: int(v[0], 16) for line in (images/'System.map').read_text().splitlines()
                if len(v := line.split()) >= 3}
        d = a.out/f'fast-{fast}-{run}-{version}'
        stage = d/'stage'; stage.mkdir(parents=True)
        profile = d/'profile'; profile.mkdir()
        arms = ['exchange', 'direct'] if run < 2 else ['direct', 'exchange']
        lines = ['#!/bin/sh', 'set -eu', "trap 'echo COPY_EXIT:$?; dmesg; echo VIRTUAL_STAGED_DONE' EXIT",
                 'cp -a /mnt/vm /tmp/helper-copy', 'cd /tmp/helper-copy', 'chmod +x *.dom',
                 '[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status',
                 'echo 0 > /proc/sys/kernel/randomize_va_space', 'killall syslogd klogd 2>/dev/null || true']
        for arm in arms:
            shutil.copy2(a.probes/f'copy-{arm}.dom', stage/f'copy-{arm}.dom')
            lines += [f'echo CASE:{arm}', f'./copy-{arm}.dom 4096', f'echo END:{arm}']
        (stage/'gate.sh').write_text('\n'.join(lines)+'\n')
        flags = ('capstone_vm.native_task=1 capstone_vm.native_io=1 capstone_vm.dom_binfmt=1 '
                 'capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1 capstone_vm.native_slot_transport=1 '
                 'capstone_vm.unpinned=1 capstone_vm.supervisor_slots=1 capstone_vm.native_route_trace=0 '
                 f'capstone_vm.native_fast={fast}')
        cmd = ['python3', str(virtual/'run-staged.py'), '--qemu', str(a.qemu), '--images', str(images),
               '--firmware', str(a.firmware), '--stage', str(stage), '--work', str(d/'vm'),
               '--disk-mib', '64', '--memory', '4G', '--timeout', '240', '--icount', '0',
               '--kernel-arg', flags, '--plugin', str(a.plugin), '--plugin-arg', 'out='+str(profile)]
        (d/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
        env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA='x-capstone-supervisor-slots=1,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true')
        with (d/'runner.log').open('w') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        serial = (d/'vm/serial.log').read_text(errors='replace').replace('\r', '')
        assert 'COPY_EXIT:0\n' in serial and 'VIRTUAL_STAGED_DONE' in serial
        assert not re.search(r'WARNING:|BUG:|Oops|Kernel panic', serial)
        assert len(list(profile.glob('window-*.tsv'))) == 2 and not list(profile.glob('*.invalid'))
        for index, arm in enumerate(arms):
            body = serial.split(f'CASE:{arm}\n', 1)[1].split(f'END:{arm}\n', 1)[0]
            assert 'MUSL_COPY bytes=1048576 chunk=4096 calls=512 failures=0\n' in body
            counts = dict(kernel=0, other=0)
            for line in (profile/f'window-{index}.tsv').read_text().splitlines():
                if line.startswith('#'): continue
                f = line.split(); pc, attempts = int(f[0], 16), int(f[3])
                counts['kernel' if syms['_stext'] <= pc < syms['_etext'] else 'other'] += attempts
            rows.append(dict(fast=fast, run=run, version=version, arm=arm, calls=512,
                             total_attempts=sum(counts.values()), counts=counts))
        print('PASS', fast, run, version, flush=True)
means = []
for fast in [0, 1]:
    for arm in ['exchange', 'direct']:
        values = {}
        for version in ['baseline', 'candidate']:
            group = [r for r in rows if r['fast'] == fast and r['arm'] == arm and r['version'] == version]
            values[version] = {kind: statistics.mean(r['counts'][kind] for r in group)/512 for kind in ['kernel', 'other']}
        before, after = values['baseline']['kernel'], values['candidate']['kernel']
        means.append(dict(fast=fast, arm=arm, attempts_per_call=values,
                          kernel_reduction_percent=100*(1-after/before)))
result = dict(status='PASS', inputs=inputs, rows=rows, means=means, limitations=[
    'System-wide attempted guest instructions inside marked read/write loops, not cycles or FPGA performance.',
    'Two repeats per kernel/route/ABI, kernel ABBA order; fixed 1 MiB resident-file copy in 4 KiB chunks.',
    'Setup and byte-for-byte output validation are outside the windows; loop/check overhead is inside.',
    'Tracing disabled. Fast=0 forces generic dispatcher; fast=1 exercises existing fast-path reuse.',
    'The outer quick gate must enforce the shared 300-second deadline. No full SPEC claim.'])
(a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
for row in means: print(json.dumps(row), flush=True)
