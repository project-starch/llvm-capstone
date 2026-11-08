#!/usr/bin/env python3
"""MEASUREMENT ONLY. One SPEC stage on the native task backend, with the delegated-call routes and the
checked-copy model selected, capmix snapshots around each command and the adapter's counters around each.

  run-native-stage.py --bench perl|gtr --arm exchange|direct --model on|off --out NEW_DIR [--plugin capmix.so]

perl = 500.perlbench_r test (makerand.pl, then test.pl driven by a plain RV64 perl that execs the native
perl for each of its 317 test files, as today's board stage does); gtr = 502.gcc_r train01.
exchange = every native route off (capstone_vm.native_io=0); direct = native_io and the seven route
features on (mask 479), the same binary in both arms. model off = x-capstone-checked-copy=false, so every
CSCHECKR/W traps to the firmware's emulation (capstone_vm.checked_copy_emu=1).
Outputs are copied to OUT/results over 9p; validation (specdiff) is a separate host step.
"""
import argparse, hashlib, json, os, shlex, shutil, subprocess, sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--bench', choices=['perl', 'gtr'], required=True)
p.add_argument('--arm', choices=['exchange', 'direct'], required=True)
p.add_argument('--model', choices=['on', 'off'], required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--qemu', type=Path, required=True)
p.add_argument('--images', type=Path, required=True)
p.add_argument('--firmware', type=Path, required=True)
p.add_argument('--dom', type=Path, required=True, help='the native-IO SDK build of the benchmark')
p.add_argument('--stage-src', type=Path, required=True, help="a make-stage.py stage (its in/, natperl, mark1/mark2)")
p.add_argument('--runner', type=Path, required=True, help='run-staged.py')
p.add_argument('--plugin', type=Path)
p.add_argument('--marks', type=Path, help='directory with the capmix marker programs mark1/mark2 (HINT 0x701/0x702), if the stage has none')
p.add_argument('--memory', default='2G')
p.add_argument('--timeout', type=int, default=21600)
p.add_argument('--tests', help='perl only: a space-separated subset of t/ files for a smoke run')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
stage, results = a.out / 'stage', a.out / 'results'
stage.mkdir(); results.mkdir()

def h(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

shutil.copytree(a.stage_src / 'in', stage / 'in', symlinks=True)
for name in ['mark1', 'mark2'] + (['natperl'] if a.bench == 'perl' else []):
    src = a.stage_src / name
    if not src.exists() and a.marks:
        src = a.marks / name
    shutil.copy2(src, stage / name)
shutil.copy2(a.dom, stage / 'bench.dom')

cnt = 'cat /sys/module/capstone_vm/parameters/native_counters'
lines = ['#!/bin/sh', 'set -u', 'R=/mnt/results', 'mkdir -p $R',
         'mount -t 9p -o trans=virtio,version=9p2000.L gate9p $R || echo @@NO9P',
         'echo 0 > /proc/sys/kernel/randomize_va_space', 'killall syslogd klogd 2>/dev/null',
         'mkdir -p /tmp/b && cp /mnt/vm/bench.dom /mnt/vm/mark1 /mnt/vm/mark2 /tmp/b/ && chmod 755 /tmp/b/*',
         'rm -rf /tmp/sr && mkdir -p /tmp/sr && cp -a /mnt/vm/in/. /tmp/sr/ && cd /tmp/sr || exit 1',
         'mkdir -m 0700 /tmp/sr/.capstone-tmp',
         'echo "@@PARAMS $(for f in native_io native_locks native_stat native_seek native_fd native_close native_path native_clock checked_copy checked_copy_emu supervisor_slots; do printf "%s=%s " $f $(cat /sys/module/capstone_vm/parameters/$f); done)"']
if a.bench == 'perl':
    lines += ['cp /mnt/vm/natperl /tmp/b/ && chmod 755 /tmp/b/natperl', 'mkdir -p /tmp/sw && ln -s /tmp/b/bench.dom /tmp/sw/perl']
    commands = [('makerand', '/tmp/b/bench.dom -I. -I./lib makerand.pl > makerand.out 2>> makerand.err')]
    if a.tests:
        driver = 'my @t = qw(' + a.tests + '); $^X="../sw/perl"; for (@t) { my $r = system($^X, "-I.", "-I./lib", $_); print "$_ rc=$r\\n"; }'
        commands.append(('tests', '/tmp/b/natperl -I. -I./lib -e ' + shlex.quote(driver) + ' > test.out 2>> test.err'))
    else:
        commands.append(('test', '/tmp/b/natperl -I. -I./lib -e \'$^X="../sw/perl"; do "./test.pl"; die $@ if $@\' > test.out 2>> test.err'))
    outputs = ['makerand.out', 'makerand.err', 'test.out', 'test.err']
else:
    commands = [('train01', '/tmp/b/bench.dom train01.c -O3 -finline-limit=50000 -o train01.opts-O3_-finline-limit_50000.s'
                 ' > train01.opts-O3_-finline-limit_50000.out 2>> train01.opts-O3_-finline-limit_50000.err')]
    outputs = ['train01.opts-O3_-finline-limit_50000.s', 'train01.opts-O3_-finline-limit_50000.out',
               'train01.opts-O3_-finline-limit_50000.err']
for i, (name, command) in enumerate(commands):
    lines += [f'echo "@@CNT {i} begin $({cnt})"', '/tmp/b/mark1', 'u0=$(cut -d" " -f1 /proc/uptime)',
              command, 'rc=$?', '/tmp/b/mark2', 'u1=$(cut -d" " -f1 /proc/uptime)',
              f'echo "@@CNT {i} end $({cnt})"', f'echo "@@RC {i} {name} rc=$rc up0=$u0 up1=$u1"']
lines += ['echo @@HASH-BEGIN', 'find . -type f | sort | while read -r f; do sha256sum "$f"; done', 'echo @@HASH-END']
lines += [f'[ -f {o} ] && cp {o} $R/' for o in outputs]
lines += ['sync', 'dmesg | grep -E "capstone|native" | tail -40', 'echo VIRTUAL_STAGED_DONE']
(stage / 'gate.sh').write_text('\n'.join(lines) + '\n')

flags = ('capstone_vm.native_task=1 capstone_vm.dom_binfmt=1 capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1 '
         'capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1 capstone_vm.supervisor_slots=1 '
         'capstone_vm.checked_copy_emu=1 capstone_vm.dom_null=1 capstone_vm.dom_local_tmp=1')
if a.arm == 'direct':
    flags += ' capstone_vm.native_io=1' + ''.join(f' capstone_vm.native_{k}=1' for k in ['locks', 'stat', 'seek', 'fd', 'close', 'path', 'clock'])
cpu = (f'x-capstone-supervisor-slots=1,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,'
       f'x-capstone-checked-copy={"true" if a.model == "on" else "false"},x-capstone-context-transport=true')
cmd = ['python3', str(a.runner), '--qemu', str(a.qemu.resolve()), '--images', str(a.images.resolve()),
       '--firmware', str(a.firmware.resolve()), '--stage', str(stage.resolve()), '--work', str((a.out / 'vm').resolve()),
       '--share', str(results.resolve()), '--memory', a.memory, '--disk-mib', '512', '--timeout', str(a.timeout),
       '--icount', '0', '--kernel-arg', flags]
if a.plugin:
    cmd += ['--plugin', str(a.plugin.resolve()), '--plugin-arg', f'out={(a.out / "capmix.txt").resolve()}', '--plugin-arg', 'shift=24']
record = dict(bench=a.bench, arm=a.arm, model=a.model, kernel_args=flags, cpu_extra=cpu, command=cmd,
              inputs={k: h(v) for k, v in [('qemu', a.qemu), ('Image', a.images / 'Image'), ('rootfs', a.images / 'rootfs.ext2'),
                                           ('firmware', a.firmware), ('dom', a.dom)] + ([('plugin', a.plugin)] if a.plugin else [])})
(a.out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA=cpu)
with (a.out / 'runner.log').open('w') as log:
    rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
print('runner rc', rc, a.out)
sys.exit(rc)
