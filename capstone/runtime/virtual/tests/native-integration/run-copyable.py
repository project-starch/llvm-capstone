#!/usr/bin/env python3
"""Copyable-only native gates on a fresh staged VM; source the test env first."""
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
virtual = here.parent.parent
p = argparse.ArgumentParser()
p.add_argument('mode', choices=['qualify', 'race', 'budget', 'admission'])
for name in ['qemu', 'images', 'module', 'probes', 'out', 'firmware']:
    p.add_argument('--'+name, type=Path, required=True)
p.add_argument('--launcher', type=Path)
p.add_argument('--plugin', type=Path)
p.add_argument('--arms', nargs='+', choices=['plain', 'launcher', 'native'], default=['plain', 'launcher', 'native'])
p.add_argument('--fast', type=int, choices=[0, 1], default=1)
p.add_argument('--features-off', action='store_true')
p.add_argument('--expect-native-refusal', action='store_true')
p.add_argument('--slots', type=int, default=32)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
stage = a.out / 'stage'
stage.mkdir()
inputs = {}
(a.out/'configuration.json').write_text(json.dumps({'mode':a.mode, 'fast':a.fast, 'slots':a.slots, 'arms':a.arms, 'repetitions':3, 'features_off':a.features_off, 'expect_native_refusal':a.expect_native_refusal}, indent=2)+'\n')
def copy(src, name=None):
    src = src.resolve()
    name = name or src.name
    shutil.copy2(src, stage / name)
    inputs[name] = hashlib.sha256(src.read_bytes()).hexdigest()
copy(a.module, 'capstone_vm.ko')
for name, src in [('qemu', a.qemu), ('Image', a.images/'Image'), ('firmware', a.firmware)]:
    inputs[name] = hashlib.sha256(src.read_bytes()).hexdigest()
if a.plugin: inputs['capprof'] = hashlib.sha256(a.plugin.read_bytes()).hexdigest()
if (a.images/'System.map').exists(): inputs['System.map'] = hashlib.sha256((a.images/'System.map').read_bytes()).hexdigest()
common = f'dom_binfmt=1 native_dispatch=1 checked_copy=1 native_slot_transport=1 supervisor_slots={a.slots} unpinned=1'
lines = ['#!/bin/sh', 'set -eu', "trap 'echo NATIVE_GATE_EXIT:$?; echo VIRTUAL_STAGED_DONE' EXIT", 'cd /mnt/vm', 'dmesg -n 1', '[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status', 'echo 0 > /proc/sys/kernel/randomize_va_space']
scenarios, arms = [], []
if a.mode == 'admission':
    lines += [f'insmod capstone_vm.ko supervisor_slots={a.slots} unpinned=1', 'rmmod capstone_vm', 'echo PASS:default-load']
    for flags in ['checked_copy=1', 'native_dispatch=1', 'native_slot_transport=1']:
        lines += [f'if insmod capstone_vm.ko {flags} unpinned=1; then rmmod capstone_vm; exit 1; fi', 'echo PASS:incomplete-profile-refused']
    if a.features_off or a.expect_native_refusal:
        lines += [f'if insmod capstone_vm.ko {common}; then rmmod capstone_vm; exit 1; fi', 'echo PASS:native-profile-refused']
    else:
        lines += [f'insmod capstone_vm.ko {common}', 'rmmod capstone_vm', 'echo PASS:native-load']
elif a.mode == 'qualify':
    copy(a.probes/'probe.dom')
    copy(a.probes/'blocked.dom')
    copy(a.probes/'entry-driver')
    lines += ['cp probe.dom /tmp/probe.dom', 'chmod +x /tmp/probe.dom', 'cp blocked.dom entry-driver /tmp/', 'chmod +x /tmp/blocked.dom /tmp/entry-driver', 'yes Q | head -c 131072 > /tmp/input']
    for fast in [0, 1]:
        lines += [f'insmod capstone_vm.ko {common} native_fast={fast}', f'echo NATIVE_BEGIN:{fast}', '(cd /tmp; ./probe.dom <input >output)', 'cat /tmp/output', '(cd /tmp; ./entry-driver ./blocked.dom)', 'echo NATIVE_FAST_HITS:$(cat /sys/module/capstone_vm/parameters/native_fast_hits)', 'rmmod capstone_vm', f'echo NATIVE_END:{fast}']
elif a.mode == 'race':
    for name in ['race.dom', 'pipe-fill']: copy(a.probes/name)
    source = here.parent/'native-race/scenarios.txt'
    shutil.copy2(source, a.out/'scenarios.txt')
    lines += [f'insmod capstone_vm.ko {common} native_fast={a.fast}', 'cp race.dom pipe-fill /tmp/', 'chmod +x /tmp/race.dom /tmp/pipe-fill']
    for row in source.read_text().splitlines():
        if not row or row.startswith('#'): continue
        name, params, args = row.split('|')
        assert re.fullmatch(r'[a-z0-9-]+', name)
        scenarios.append(name)
        for option in ['rt_norecheck', 'rt_noexcl', 'rt_dummy', 'rt_nohold']:
            lines += [f'echo {int(option in params.split())} > /sys/module/capstone_vm/parameters/{option}']
        argv = ' '.join(shlex.quote(v) for v in args.split())
        lines += [f'echo NATIVE_CASE:{name}', 'status=0', f'(cd /tmp; ./pipe-fill ./race.dom {argv}) || status=$?', 'echo RC=$status', f'echo NATIVE_END:{name}']
    lines += ['rmmod capstone_vm']
else:
    if not a.plugin or not a.launcher: p.error('budget needs --plugin and --launcher')
    for name in ['pipe-exec', 'budget-plain', 'budget-legacy.dom', 'budget-native.dom']: copy(a.probes/name)
    copy(a.launcher, 'launcher')
    lines += ['cp pipe-exec budget-* launcher /tmp/', 'chmod +x /tmp/pipe-exec /tmp/budget-* /tmp/launcher', 'killall syslogd klogd 2>/dev/null || true']
    # Alternate order across repetitions, separate module modes so the old
    # launcher retains its ordinary child-frame/slot contract.
    for rep in range(1, 4):
        for arm in (['plain', 'launcher', 'native'] if rep % 2 else ['native', 'launcher', 'plain']):
            if arm not in a.arms: continue
            config = common+f' native_fast={a.fast}' if arm == 'native' else f'supervisor_slots={a.slots} unpinned=1'
            lines += [f'insmod capstone_vm.ko {config}']
            command = {'plain': './budget-plain', 'launcher': './launcher ./budget-legacy.dom', 'native': './budget-native.dom'}[arm]
            for size, direction in [(0, "read"), (1, "read"), (65536, "read"), (1, "write"), (65536, "write")]:
                index = len(arms)
                arms.append((index, arm, size, rep, direction))
                verify = f'test "$(head -c {65536 if size == 65536 else 100} budget.out | tr -d X | wc -c)" -eq 0; ' if direction == 'write' else ''
                lines += [f'echo NATIVE_CASE:{index}', 'for f in /sys/module/capstone_vm/sections/.*; do [ ! -f "$f" ] || echo MODULE_SECTION:${f##*/}:$(cat "$f"); done', f'(cd /tmp; ./pipe-exec {command} {size} {direction} >budget.out; {verify}tail -c 96 budget.out)', f'echo NATIVE_END:{index}']
            lines += ['rmmod capstone_vm']
    (a.out/'arms.tsv').write_text(''.join('\t'.join(map(str, r))+'\n' for r in arms))
lines += ['echo NATIVE_UNLOAD_PASS', 'dmesg']
(stage/'gate.sh').write_text('\n'.join(lines)+'\n')
(stage/'gate.sh').chmod(0o755)
(a.out/'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
env = dict(os.environ)
env['CAPSTONE_QEMU_CPU_EXTRA'] = f'x-capstone-supervisor-slots={a.slots},x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true'
if a.features_off:
    env['CAPSTONE_QEMU_CPU_EXTRA'] = env['CAPSTONE_QEMU_CPU_EXTRA'].replace('x-capstone-checked-copy=true', 'x-capstone-checked-copy=false').replace('x-capstone-context-transport=true', 'x-capstone-context-transport=false')
cmd = ['python3', str(virtual/'run-staged.py'), '--qemu', str(a.qemu.resolve()), '--images', str(a.images.resolve()), '--firmware', str(a.firmware.resolve()), '--stage', str(stage.resolve()), '--work', str((a.out/'vm').resolve()), '--disk-mib', '96', '--timeout', '360']
if a.mode == 'budget':
    (a.out/'capprof').mkdir()
    cmd += ['--icount', '0', '--plugin', str(a.plugin.resolve()), '--plugin-arg', 'out='+str((a.out/'capprof').resolve())]
with (a.out/'run.log').open('w') as log: subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
log = (a.out/'vm/serial.log').read_text(errors='replace').replace('\r', '')
assert 'NATIVE_GATE_EXIT:0\n' in log and 'NATIVE_UNLOAD_PASS\n' in log, log[-6000:]
assert not re.search(r'WARNING:|BUG:|Oops|Unable to handle|Kernel panic', log), 'kernel diagnostic'
if a.mode == 'admission':
    assert log.count('PASS:') == 5, log
elif a.mode == 'qualify':
    assert log.count('native integration checks=51 failures=0') == 2, log
    assert log.count('native entry failures=0') == 2, log
    assert 'NATIVE_FAST_HITS:0\n' in log and re.search(r'NATIVE_FAST_HITS:[1-9][0-9]*\n', log)
else:
    for name in scenarios or [str(r[0]) for r in arms]:
        body = log.split('NATIVE_CASE:'+name+'\n', 1)[1].split('NATIVE_END:'+name+'\n', 1)[0]
        (a.out/(name+'.log')).write_text(body)
    oracle = here.parent/'native-race/judge.py' if a.mode == 'race' else here/'summarize.py'
    oracle_cmd = ['python3', str(oracle), str(a.out)]
    if a.mode == 'budget' and (a.images/'System.map').exists():
        oracle_cmd += ['--kernel-map', str(a.images/'System.map')]
    with (a.out/'verdict.txt').open('w') as out: subprocess.run(oracle_cmd, stdout=out, check=True)
if a.mode in ('qualify', 'admission'):
    (a.out/'verdict.txt').write_text('PASS '+a.mode+'\n')
print('PASS', a.mode, a.out)
