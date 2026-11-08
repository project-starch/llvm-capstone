#!/usr/bin/env python3
"""MEASUREMENT ONLY. One SPEC stage on the native task backend, with the delegated-call routes and the
checked-copy model selected, capmix snapshots around each command and the adapter's counters around each.

  run-native-stage.py --bench perl|gtr --arm exchange|direct --model on|off --out NEW_DIR [--plugin capmix.so]

perl = 500.perlbench_r test, SPEC's two invocations (makerand.pl, test.pl) by the native perl itself, run from
the staged directory as ./perlbench_r; test.pl's children start through native spawn and the pinned native hush
shell (--shell), with null-device and private-/tmp grants, as the collaborator's qualified Perl run. (Today's
board stage drives test.pl with a plain RV64 perl instead; on the native backend that cannot work: an exec's
grant root is its cwd, and t/TEST execs each test from t/ with -I../lib.) gtr = 502.gcc_r train01.
exchange = every native route off (capstone_vm.native_io=0); direct = native_io and the seven route
features on (mask 479), the same binary in both arms. model off = x-capstone-checked-copy=false, so every
CSCHECKR/W traps to the firmware's emulation (capstone_vm.checked_copy_emu=1).
Outputs are copied to OUT/results over 9p; validation (specdiff) is a separate host step.
"""
import argparse, hashlib, json, os, shlex, shutil, subprocess, sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--bench', choices=['perl', 'gtr', 'plan'], required=True)
p.add_argument('--plan', type=Path, help='plan mode: a make-stage.py stage directory (plan.json, stage/in)')
p.add_argument('--exe', action='append', default=[], help='plan mode: NAME=PATH, the native build of NAME_base.ARM')
p.add_argument('--arm', choices=['exchange', 'direct'], required=True)
p.add_argument('--model', choices=['on', 'off'], required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--qemu', type=Path, required=True)
p.add_argument('--images', type=Path, required=True)
p.add_argument('--firmware', type=Path, required=True)
p.add_argument('--dom', type=Path, help='the native-IO SDK build of the benchmark (perl, gtr)')
p.add_argument('--stage-src', type=Path, required=True, help="a make-stage.py stage (its in/, natperl, mark1/mark2)")
p.add_argument('--runner', type=Path, required=True, help='run-staged.py')
p.add_argument('--plugin', type=Path)
p.add_argument('--shell', type=Path, help='perl: the native hush.dom for test.pl children')
p.add_argument('--c5a-model', action='store_true', help="C.5a's model: no transport, no raw moves, pinned arenas, the adapter's c5a_model")
p.add_argument('--twin', action='store_true', help="the C.5a twin QEMU (flat-index-20 + the frame-semantics check model): its own -cpu properties, the C.5a device tree")
p.add_argument('--dtb', type=Path)
p.add_argument('--transport', action='store_true', help='c5a: use the slot detach/adopt (CSRUNV 4/5)')
p.add_argument('--marks', type=Path, help='directory with the capmix marker programs mark1/mark2 (HINT 0x701/0x702), if the stage has none')
p.add_argument('--memory', default='2G')
p.add_argument('--timeout', type=int, default=21600)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
stage, results = a.out / 'stage', a.out / 'results'
stage.mkdir(); results.mkdir()

def h(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if a.bench == 'plan':
    a.stage_src = a.plan / 'stage'
    plan = json.loads((a.plan / 'plan.json').read_text())
shutil.copytree(a.stage_src / 'in', stage / 'in', symlinks=True)
for name in ['mark1', 'mark2']:
    src = a.stage_src / name
    if not src.exists() and a.marks:
        src = a.marks / name
    shutil.copy2(src, stage / name)
if a.bench == 'plan':
    exes = dict(x.split('=', 1) for x in a.exe)
    for name, path in exes.items():
        shutil.copy2(path, stage / (name + '.dom'))
    for extra in ['ldecod']:
        if (a.stage_src / extra).exists():
            shutil.copy2(a.stage_src / extra, stage / extra)
else:
    shutil.copy2(a.dom, stage / 'bench.dom')
if a.bench == 'perl':
    shutil.copy2(a.shell, stage / 'hush.dom')

cnt = 'cat /sys/module/capstone_vm/parameters/native_counters'
lines = ['#!/bin/sh', 'set -u', 'R=/mnt/results', 'mkdir -p $R',
         'mount -t 9p -o trans=virtio,version=9p2000.L gate9p $R || echo @@NO9P',
         'echo 0 > /proc/sys/kernel/randomize_va_space', 'killall syslogd klogd 2>/dev/null',
         'mkdir -p /tmp/b && cp /mnt/vm/bench.dom /mnt/vm/mark1 /mnt/vm/mark2 /tmp/b/ && chmod 755 /tmp/b/*',
         'rm -rf /tmp/sr && mkdir -p /tmp/sr && cp -a /mnt/vm/in/. /tmp/sr/ && cd /tmp/sr || exit 1',
         'mkdir -m 0700 /tmp/sr/.capstone-tmp',
         'echo "@@PARAMS $(for f in native_io native_locks native_stat native_seek native_fd native_close native_path native_clock checked_copy checked_copy_emu supervisor_slots; do printf "%s=%s " $f $(cat /sys/module/capstone_vm/parameters/$f); done)"']
if a.bench == 'perl':
    lines += ['cp /mnt/vm/bench.dom /tmp/sr/perlbench_r && chmod 755 /tmp/sr/perlbench_r',
              'mkdir -p /tmp/s && cp /mnt/vm/hush.dom /tmp/s/h.dom && chmod 755 /tmp/s/h.dom']
    commands = [('makerand', './perlbench_r -I. -I./lib makerand.pl > makerand.out 2>> makerand.err'),
                ('test', './perlbench_r -I. -I./lib test.pl > test.out 2>> test.err')]
    outputs = ['makerand.out', 'makerand.err', 'test.out', 'test.err']
elif a.bench == 'plan':
    lines += ['cp /mnt/vm/*.dom /tmp/b/ && chmod 755 /tmp/b/*.dom', '[ -f /mnt/vm/ldecod ] && cp /mnt/vm/ldecod /tmp/b/ && chmod 755 /tmp/b/ldecod']
    commands, outputs = [], []
    for kind in ['generation', 'commands']:
        for i, c in enumerate(plan[kind]):
            argv = list(c['argv'])
            exe = argv[0]
            if exe.endswith('_base.ARM'):
                argv[0] = '/tmp/b/' + exe[:-len('_base.ARM')] + '.dom'
            else:
                argv[0] = '/tmp/b/' + exe.split('_')[0]
            line = ' '.join(shlex.quote(x) for x in argv)
            if c.get('input'): line += ' < ' + shlex.quote(c['input'])
            if c.get('output'): line += ' > ' + shlex.quote(c['output']); outputs.append(c['output'])
            if c.get('error'): line += ' 2>> ' + shlex.quote(c['error']); outputs.append(c['error'])
            commands.append((('gen' if kind == 'generation' else 'run') + str(i), line))
    for v in plan['host_validation']:
        outputs.append(v['argv'][-1])
    outputs = sorted(set(outputs))
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
         + ('capstone_vm.c5a_model=1 ' if a.c5a_model else 'capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1 ')
         + 'capstone_vm.checked_copy_emu=1 capstone_vm.dom_null=1 capstone_vm.dom_local_tmp=1')
if a.bench == 'perl':
    flags += ' capstone_vm.dom_shell=/tmp/s/h.dom'
if a.arm == 'direct':
    flags += ' capstone_vm.native_io=1' + ''.join(f' capstone_vm.native_{k}=1' for k in ['locks', 'stat', 'seek', 'fd', 'close', 'path', 'clock'])
# supervisor_slots=1 is the adapter's default; the kernel keeps 511 command-line bytes, so it is not spelled out.
if a.twin:
    flags = flags.replace('capstone_vm.dom_null=1 ', '') if a.bench != 'perl' else flags
    flags += ' cma=256M' + (' capstone_vm.native_slot_transport=1' if a.transport else '')
assert len(('root=/dev/vda ro ' + flags).encode()) < 512, 'kernel command line would be truncated'
cpu = (f'x-capstone-supervisor-slots=1,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,'
       f'x-capstone-checked-copy={"true" if a.model == "on" else "false"},x-capstone-context-transport=true')
if a.twin:
    cpu = 'x-capstone-supervisor-slots=1,svadu=false,x-capstone-cscheck-frame=' + ('true' if a.model == 'on' else 'false')
elif a.c5a_model:
    cpu = cpu.replace('x-capstone-raw-moves=true', 'x-capstone-raw-moves=false').replace('x-capstone-context-transport=true', 'x-capstone-context-transport=false')
cmd = ['python3', str(a.runner), '--qemu', str(a.qemu.resolve()), '--images', str(a.images.resolve()),
       '--firmware', str(a.firmware.resolve()), '--stage', str(stage.resolve()), '--work', str((a.out / 'vm').resolve()),
       '--share', str(results.resolve()), '--memory', a.memory, '--disk-mib', '512', '--timeout', str(a.timeout),
       '--icount', '0', '--kernel-arg', flags] + (['--dtb', str(a.dtb.resolve())] if a.dtb else [])
record_exes = {x.split('=', 1)[0]: h(x.split('=', 1)[1]) for x in a.exe}
if a.plugin:
    cmd += ['--plugin', str(a.plugin.resolve()), '--plugin-arg', f'out={(a.out / "capmix.txt").resolve()}', '--plugin-arg', 'shift=24']
record = dict(exes=record_exes, plan=str(a.plan) if a.plan else None, bench=a.bench, arm=a.arm, model=a.model, kernel_args=flags, cpu_extra=cpu, command=cmd,
              inputs={k: h(v) for k, v in [('qemu', a.qemu), ('Image', a.images / 'Image'), ('rootfs', a.images / 'rootfs.ext2'),
                                           ('firmware', a.firmware)] + ([('dom', a.dom)] if a.dom else []) + ([('shell', a.shell)] if a.bench == 'perl' else []) + ([('plugin', a.plugin)] if a.plugin else [])})
(a.out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA=cpu)
with (a.out / 'runner.log').open('w') as log:
    rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
print('runner rc', rc, a.out)
sys.exit(rc)
