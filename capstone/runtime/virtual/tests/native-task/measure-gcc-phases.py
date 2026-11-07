#!/usr/bin/env python3
"""Matched plain/native GCC phases on one built-in kernel; no module loader."""
import argparse
import bisect
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
for name in ['qemu', 'images', 'firmware', 'plugin', 'profrun', 'gcc', 'plain', 'input', 'reference', 'out']:
    p.add_argument('--'+name, type=Path, required=True)
p.add_argument('--resolver', type=Path, default=Path(__file__).resolve().parents[4]/'ports/spec-cpu2017/tools/stop-at.py')
p.add_argument('--negative-boundary', action='store_true')
p.add_argument('--arm', action='append', choices=['plain', 'native'], help='Override run order; repeat to specify multiple runs.')
p.add_argument('--kernel-arg', action='append', default=[], help='Additional kernel argument applied to both arms (diagnostic experiments).')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
inputs = {n: sha(getattr(a, n)) for n in ['qemu', 'firmware', 'plugin', 'profrun', 'gcc', 'plain', 'input', 'reference', 'resolver']}
inputs.update({n: sha(a.images/n) for n in ['Image', 'System.map', 'rootfs.ext2']})
syms = {v[2]: int(v[0], 16) for line in (a.images/'System.map').read_text().splitlines() if len(v := line.split()) >= 3}
ks = sorted((address, name) for name, address in syms.items() if syms['_stext'] <= address < syms['_etext'])
ka = [x[0] for x in ks]
rows = []
phases = ['startup_and_initialization', 'compilation_and_output', 'finalization_and_exit']
for index, arm in enumerate(['native'] if a.negative_boundary else (a.arm or ['plain', 'native', 'native', 'plain'])):
    d = a.out/f'{index}-{arm}'
    stage = d/'stage'; stage.mkdir(parents=True)
    (d/'profile').mkdir()
    exe = a.plain if arm == 'plain' else a.gcc
    for src, name in [(exe, 'gcc.bin'), (a.profrun, 'profrun'), (a.input, 't1.c'), (a.reference, 'reference.s')]:
        shutil.copy2(src, stage/name)
    (stage/'gate.sh').write_text('''#!/bin/sh
set -eu
trap 'echo GCC_PHASE_EXIT:$?; echo VIRTUAL_STAGED_DONE' EXIT
cp -a /mnt/vm /tmp/gcc-phase
cd /tmp/gcc-phase
[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status
echo 0 > /proc/sys/kernel/randomize_va_space
killall syslogd klogd 2>/dev/null || true
chmod +x profrun gcc.bin
counters() {
 for f in /sys/module/capstone_vm/parameters/unpinned_faults /sys/module/capstone_vm/parameters/unpinned_fast_faults /sys/module/capstone_vm/parameters/unpinned_sweep_pages /sys/module/capstone_vm/parameters/fault_ahead_pages /sys/module/capstone_vm/parameters/fault_ahead_windows /sys/module/*/parameters/cleared_pages /sys/module/*/parameters/freed_pages /sys/module/*/parameters/copies; do
  [ ! -f "$f" ] || echo COUNTER "$1" "$f" "$(cat "$f")"
 done
}
counters before
./profrun ./gcc.bin t1.c -O3 -finline-limit=50000 -o output.s
counters after
cmp output.s reference.s
echo PASS:GCC_EXACT_OUTPUT
dmesg
''')
    sites = json.loads(subprocess.check_output(['python3', str(a.resolver), 'resolve', '--nm', 'nm', '--kind', 'plain' if arm == 'plain' else 'capstone', '--exe', str(exe), '--symbol', 'statistics_fini', '--count', '1', '--start-symbol', 'init_cgraph', '--start-count', '1'], text=True))
    if a.negative_boundary:
        sites['plugin_args'] = [('start_bytes='+format(int(v.split('=')[1], 16)^0x10, 'x')) if v.startswith('start_bytes=') else v for v in sites['plugin_args']]
    (d/'sites.json').write_text(json.dumps(sites, indent=2)+'\n')
    flags = ' '.join('capstone_vm.'+s for s in ['native_task=1', 'native_io=1', 'native_locks=1', 'native_stat=1', 'native_seek=1', 'native_fd=1', 'native_close=1', 'native_path=1', 'native_clock=1', 'native_fast=1', 'dom_binfmt=1', 'native_dispatch=1', 'checked_copy=1', 'native_slot_transport=1', 'unpinned=1', 'supervisor_slots=32'])
    cmd = ['python3', str(Path(__file__).resolve().parents[2]/'run-staged.py'), '--qemu', str(a.qemu), '--images', str(a.images), '--firmware', str(a.firmware), '--stage', str(stage), '--work', str(d/'vm'), '--memory', '4G', '--disk-mib', '128', '--timeout', '180', '--icount', '0', '--kernel-arg', flags, '--plugin', str(a.plugin), '--plugin-arg', 'out='+str(d/'profile')]
    for arg in a.kernel_arg: cmd += ['--kernel-arg', arg]
    for arg in ['phases=1']+sites['plugin_args']: cmd += ['--plugin-arg', arg]
    (d/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA='x-capstone-supervisor-slots=32,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true')
    with (d/'runner.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (d/'vm/serial.log').read_text(errors='replace').replace('\r', '')
    assert 'GCC_PHASE_EXIT:0\n' in log and 'PASS:GCC_EXACT_OUTPUT\n' in log
    assert not re.search(r'WARNING:|BUG:|Oops|Kernel panic', log)
    invalid = list((d/'profile').glob('*.invalid'))
    if a.negative_boundary:
        assert invalid, 'corrupt boundary was not rejected'
        (a.out/'summary.json').write_text(json.dumps(dict(status='PASS', negative_control=True, kernel_args=a.kernel_arg, inputs=inputs, invalid=[x.read_text() for x in invalid]), indent=2)+'\n')
        print('PASS corrupted boundary rejected; output still correct', flush=True)
        break
    assert not invalid and len(list((d/'profile').glob('window-*.tsv'))) == 3
    headers = ['\n'.join(x for x in (d/f'profile/window-{i}.tsv').read_text().splitlines() if x.startswith('#')) for i in range(3)]
    for i, label in [(0, 'started_at'), (1, 'stopped_at')]:
        assert re.search(r'^# '+label+r'=[0-9a-f]+:1 .*anchor_hits=[1-9][0-9]* broken=0 armed_bases=1$', headers[i], re.M), headers[i]
        which = 'start' if i == 0 else 'stop'
        candidates = re.search(r'^# '+which+r'_candidates (.*)$', headers[i], re.M)[1].split()
        used = [v.split(':') for v in candidates if int(v.split(':')[2])]
        assert len(used) == 1 and used[0][2:] == ['1', '1', '0'], candidates
    base = int(re.search(r' base=([0-9a-f]+)', headers[0])[1], 16) if arm == 'native' else 0
    ps = []
    for line in subprocess.check_output(['nm', '-n', '-S', '--defined-only', str(exe)], text=True).splitlines():
        v = line.split()
        if len(v) == 4 and v[2] in 'TtWw': ps.append((int(v[0], 16), int(v[1], 16), v[3]))
    ps.sort(); pa = [x[0] for x in ps]
    totals = []
    for i, phase in enumerate(phases):
        counts = Counter(); functions = Counter(); entries = Counter()
        for line in (d/f'profile/window-{i}.tsv').read_text().splitlines():
            if line.startswith('#'): continue
            v = line.split(); pc, n = int(v[0], 16), int(v[3])
            if syms['_stext'] <= pc < syms['_etext']:
                address, name = ks[bisect.bisect_right(ka, pc)-1]
                key = 'kernel:'+name; category = 'kernel'
                if pc == address: entries[key] += n
            else:
                link = pc-base+sites['low'] if arm == 'native' else pc
                j = bisect.bisect_right(pa, link)-1
                if j >= 0 and ps[j][0] <= link < ps[j][0]+ps[j][1]:
                    key = 'program:'+ps[j][2]; category = 'program'
                    if link == ps[j][0]: entries[key] += n
                else: key = 'other'; category = 'other'
            counts[category] += n; functions[key] += n
        total = sum(counts.values()); totals.append(total)
        assert total == int(re.search(r'^# total (\d+)', headers[i], re.M)[1])
        (d/f'{phase}-symbols.json').write_text(json.dumps(dict(attempts=functions, entry_attempts=entries), indent=2)+'\n')
        row = dict(run=index, arm=arm, phase=phase, total_attempts=total, classes=dict(counts), reference_match=True)
        rows.append(row); print(json.dumps(row), flush=True)
    outer = int((d/'profile/phase-total.txt').read_text()); assert sum(totals) == outer
    counters = {when: {name:int(value) for w,name,value in re.findall(r'^COUNTER (before|after) (\S+) (\d+)$', log, re.M) if w == when} for when in ['before', 'after']}
    delta = {k:v-counters['before'][k] for k,v in counters['after'].items() if k in counters['before']}
    (d/'verdict.json').write_text(json.dumps(dict(status='PASS', phase_sum=sum(totals), outer_attempts=outer, counter_deltas=delta, reference_match=True), indent=2)+'\n')
if not a.negative_boundary:
    (a.out/'summary.json').write_text(json.dumps(dict(status='PASS', kernel_args=a.kernel_arg, inputs=inputs, rows=rows, boundaries=['first init_cgraph entry', 'first statistics_fini entry'], limitations=['QEMU attempted instructions, not hardware cycles or elapsed time.', 'Ordinary RV64 and Capstone builds have different recorded compiler/libc provenance; not pure ISA isolation.', 'Same patched Linux/model, not stock-kernel comparison.', 'Startup includes GCC backend initialization; compilation includes output close; finalization includes process exit.', 'Flat symbols and first-instruction attempts are not inclusive costs or guaranteed invocation counts.', 'Sysfs counter deltas cover invocation plus measurement shell activity; compare with the plain arm.']), indent=2)+'\n')
    print('PASS exact outputs, authenticated boundaries and phase conservation', flush=True)
