#!/usr/bin/env python3
"""Replay census4 test invocations in private run directories, preserving SPEC oracles.

Run on p12 after sourcing capstone-test-env.sh. No SPEC source or shared run
is modified. Uses the already built census4/rv64 images and input data.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys

W = Path(os.environ.get('LANE_ROOT', str(Path.home() / 'lanes/laneAG2')))
BENCH, ARM = sys.argv[1:]
if ARM not in ('plain', 'launcher', 'native'):
    sys.exit('arm must be plain, launcher or native')
PL = Path(os.environ.get('NATIVE_PLATFORM', str(W / 'platform-native')))
OUT = Path(os.environ.get('SPEC_RUN_ROOT', str(W / 'runs'))) / ARM / BENCH
SPEC = Path.home() / 'prof-dev/tree/SPEC17'
CENSUS = Path.home() / 'prof-dev/runs/census4' / ('rv64' if ARM == 'plain' else 'cap-dev4') / BENCH
NM = Path.home() / 'prof-dev/cap4/build/bin/llvm-nm'
KIT = Path.home() / 'prof-dev/kit4'
print('free_bytes', shutil.disk_usage(W).free, flush=True)
if shutil.disk_usage(W).free < 12 * 1024**3:
    sys.exit('UNRESOLVED: insufficient disk')
OUT.mkdir(parents=True, exist_ok=False)
for name in ('invocations', 'capprof', 'work', 'profile'):
    (OUT / name).mkdir()
env = dict(os.environ, PYTHONPATH=str(W / 'src/capstone/runtime/host'),
           PYTHONDONTWRITEBYTECODE='1', CAPSTONE_QEMU_LOCK=str(OUT / 'qemu.lock'))
VM = [sys.executable, '-m', 'capstone_vm', '--state', str(OUT / 'vm')]

def vm(*args, timeout=1800):
    return subprocess.run(VM + list(map(str, args)), env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)

def must(*args):
    r = vm(*args)
    if r.returncode:
        raise RuntimeError(r.stdout[-3000:])
    return r.stdout

def guest(path):
    return '/mnt/host/' + str(path.relative_to(Path.home()))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

records = [json.loads(p.read_text()) for p in sorted((CENSUS / 'invocations').glob('[0-9][0-9][0-9][0-9].json'))]
records = [r for r in records if r.get('route') in ('guest', 'guest-plain', 'host-spec-tool')]
expected = sum(r['route'] != 'host-spec-tool' for r in records)
expected_oracles = sum(r['route'] == 'host-spec-tool' for r in records)
if not expected or not expected_oracles or any(
        r.get('rc', r.get('status')) != 0 or
        ('rc' in r and 'status' in r and r['rc'] != r['status']) for r in records):
    sys.exit('UNRESOLVED: census must contain successful invocations and SPEC oracles')
for group in ('all', 'test'):
    inputs = SPEC / 'benchspec/CPU' / BENCH / 'data' / group / 'input'
    if inputs.exists():
        shutil.copytree(inputs, OUT / 'work', dirs_exist_ok=True)
output_names = {'502.gcc_r': ['t1.opts-O3_-finline-limit_50000.out'],
                '538.imagick_r': ['test_convert.out', 'test_validate.out'],
                '519.lbm_r': ['lbm.out'], '544.nab_r': ['hkrdenq.out']}[BENCH]
if expected != len(output_names):
    sys.exit('UNRESOLVED: census invocation count changed')
inputs = {str(p): sha(p) for p in (PL / 'adapter/module/capstone_vm.ko',
          PL / 'adapter/capstone-vexec', PL / 'qemu-system-riscv64', PL / 'images/Image',
          PL / 'images/fw_jump.elf', PL / 'plugins/libcapprof.so', W / 'trampoline.bin')}
(OUT / 'inputs.json').write_text(json.dumps(inputs, indent=2))
checks = []
n = 0
oracles = 0
native = int(ARM == 'native')
up = vm('up', '--profile', 'virtual', '--memory', '8G', '--share-model', 'mapped-xattr',
        '--share', Path.home(), '--no-pcrel', '--icount', '0', '--exact-bounds',
        '--cpu-prop', 'x-capstone-test-csrs=true', '--cpu-prop', 'x-capstone-supervisor-slots=2',
        '--plugin', PL / 'plugins/libcapprof.so', '--plugin-arg', 'out=' + str(OUT / 'capprof'),
        '--qemu', PL / 'qemu-system-riscv64', '--kernel', PL / 'images/Image',
        '--firmware', PL / 'images/fw_jump.elf', '--rootfs', PL / 'images/rootfs.ext2',
        '--launcher', PL / 'adapter/capstone-vexec', '--job-helper', PL / 'adapter/capstone-job',
        '--module', PL / 'adapter/module/capstone_vm.ko', '--ssh-server', PL / 'ssh-server/dropbearmulti')
(OUT / 'up.log').write_text(up.stdout)
try:
    if up.returncode: raise RuntimeError('VM boot failed')
    if native:
        must('exec', '--', 'sh', '-c',
             'if [ -e /proc/sys/fs/binfmt_misc/status ]; then '
             'echo 0 >/proc/sys/fs/binfmt_misc/status; fi')
    must('exec', '--', 'sh', '-c', f'echo 0 >/proc/sys/kernel/randomize_va_space; '
         f'killall syslogd klogd 2>/dev/null; rmmod capstone_vm && '
         f'insmod /mnt/control/module dom_binfmt={native} supervisor_slots=2 && '
         f'cp {shlex.quote(guest(PL / "profrun"))} /tmp/profrun && chmod +x /tmp/profrun')
    sections = must('exec', '--', 'sh', '-c',
        'cd /sys/module/capstone_vm/sections && for f in .* *; do '
        '[ -f "$f" ] && echo "$f $(cat "$f")"; done')
    (OUT / 'module-sections.txt').write_text(sections)
    for r in records:
        if r['route'] == 'host-spec-tool':
            penv = dict(os.environ, SPEC=str(SPEC), SPECPERLLIB=str(SPEC / 'bin/lib') + ':' + str(SPEC / 'bin'))
            q = subprocess.run(r['argv'], cwd=OUT / 'work', env=penv, capture_output=True, text=True)
            (OUT / f'oracle-{oracles}.log').write_text(q.stdout + q.stderr)
            checks.append({'oracle': r['argv'], 'rc': q.returncode})
            oracles += 1
            continue
        exe = Path(r['exe'])
        if sha(exe) != r['exe_sha256']: raise RuntimeError('census executable identity changed')
        ident = f'{n:04d}'
        must('exec', '--', 'sh', '-c', f'cp {shlex.quote(guest(exe))} /tmp/program && chmod +x /tmp/program')
        args = []
        staged = []
        for a in r['argv'][1:]:
            if a.startswith(str(SPEC) + '/'):
                # Some SPEC validators receive absolute reference-input paths.
                # Copy only that immutable input into the explicit cwd grant;
                # preserve its bytes/hash and record the argv translation.
                source = Path(a).resolve(strict=True)
                source.relative_to((SPEC / 'benchspec/CPU' / BENCH / 'data').resolve())
                target = OUT / 'work' / ('.reference-' + str(len(staged)) + '-' + source.name)
                shutil.copyfile(source, target)
                if sha(source) != sha(target): raise RuntimeError('reference input changed')
                args.append(guest(target))
                staged.append({'source': str(source), 'target': str(target), 'sha256': sha(target)})
            else:
                args.append(guest(Path(a)) if a.startswith(str(Path.home()) + '/') else a)
        command = ['/tmp/program', *args]
        if ARM == 'launcher': command = ['/usr/bin/capstone-exec', '--', *command]
        runenv = ['env', 'OMP_NUM_THREADS=1', 'OMP_THREAD_LIMIT=1']
        if ARM == 'launcher': runenv += ['CAPSTONE_PROFILE_RECORD=/tmp/image-record', 'CAPSTONE_VM_STATS=1']
        command = f'cd {shlex.quote(guest(OUT / "work"))} && exec ' + shlex.join(runenv + ['/tmp/profrun'] + command)
        command += ' >/tmp/program.out 2>/tmp/program.err'
        q = vm('exec', '--', 'sh', '-c', command)
        checks.append({'invocation': n, 'rc': q.returncode})
        (OUT / 'invocations' / f'{ident}.vm.log').write_text(q.stdout)
        must('exec', '--', 'sh', '-c', f'cp /tmp/program.out {shlex.quote(guest(OUT / "work" / output_names[n]))} && '
             f'cp /tmp/program.err {shlex.quote(guest(OUT / "invocations" / (ident + ".err")))}')
        if q.returncode:
            print(BENCH, ARM, 'invocation', n, 'rc', q.returncode, flush=True)
            break
        if ARM == 'native':
            addresses = must('exec', '--', 'sh', '-c',
                'cat /sys/module/capstone_vm/parameters/dom_last_base /sys/module/capstone_vm/parameters/dom_last_trampoline')
            (OUT / 'invocations' / f'{ident}.addresses').write_text(addresses)
            base, tramp = map(int, addresses.split())
            syms = subprocess.check_output([NM, '--defined-only', exe], text=True)
            main = next(int(l.split()[0], 16) for l in syms.splitlines() if l.split()[-1] == 'domain_main')
            b = exe.read_bytes()
            phoff = struct.unpack_from('<Q', b, 32)[0]
            phsize, phnum = struct.unpack_from('<HH', b, 54)
            text = next(struct.unpack_from('<IIQQQQQQ', b, phoff + phsize * i)
                        for i in range(phnum) if struct.unpack_from('<II', b, phoff + phsize * i) == (1, 5))
            line = f'capstone-exec: domain image entry=0x{base+main-text[3]:x} code=0x{base:x}-0x{base+text[6]:x}\n'
        elif ARM == 'launcher':
            line = must('exec', '--', 'cat', '/tmp/image-record')
            must('exec', '--', 'rm', '-f', '/tmp/image-record')
        else: line = ''
        with (OUT / 'image-records.txt').open('a') as f: f.write(line)
        rec = dict(route='guest-plain' if ARM == 'plain' else 'guest', exe=str(exe),
                   exe_sha256=sha(exe), argv=r['argv'], guest_command=command,
                   staged_inputs=staged, rc=q.returncode, detail={'window': n})
        (OUT / 'invocations' / f'{ident}.json').write_text(json.dumps(rec, indent=2))
        print(BENCH, ARM, 'invocation', n, 'rc', q.returncode, flush=True)
        n += 1
        if q.returncode: break
    (OUT / 'dmesg.log').write_text(must('exec', '--', 'dmesg'))
except Exception as error:
    checks.append({'harness_error': str(error), 'rc': 1})
    print(BENCH, ARM, 'harness_error', str(error), flush=True)
finally:
    vm('down')
# An incomplete invocation list or missing/malformed window is a failure.
checks.append({'invocations_complete': n == expected})
checks.append({'oracles_complete': oracles == expected_oracles})
for i in range(n):
    p = OUT / 'capprof' / f'window-{i}.tsv'
    lines = p.read_text().splitlines() if p.exists() else []
    complete = bool(lines and lines[-1].startswith('# total '))
    complete = complete and sum(int(x.split('\t')[3]) for x in lines if not x.startswith('#')) == int(lines[-1].split()[2])
    checks.append({'window_complete': complete})
ok = n == expected and oracles == expected_oracles and all(
    c.get('rc', 0) == 0 and c.get('window_complete', True) for c in checks)
result = {'verdict': 'PASS' if ok else 'FAIL', 'arm': ARM, 'checks': checks}
(OUT / 'result.json').write_text(json.dumps(result, indent=2))
print(BENCH, ARM, 'verdict=' + result['verdict'], flush=True)
if ok:
    penv = dict(env, ORIGINAL_PROF_REPORT=str(KIT / 'tools/prof-report.py'), DOM_SPEC_NATIVE=str(native),
                DOM_TRAMPOLINE=str(W / 'trampoline.bin'))
    with (OUT / 'profile/summary.txt').open('w') as log:
        subprocess.run([sys.executable, str(Path(__file__).with_name('spec-profile.py')), '--out', str(OUT),
                        '--platform', str(PL), '--kind', 'plain' if ARM == 'plain' else 'capstone', '--nm', str(NM)],
                       env=penv, stdout=log, stderr=subprocess.STDOUT, check=True)
sys.exit(0 if ok else 1)
