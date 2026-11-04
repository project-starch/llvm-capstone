#!/usr/bin/env python3
"""Run exchange I/O tests in one private guest; expected refusals are explicit."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--platform', type=Path, required=True)
p.add_argument('--host', type=Path, required=True)
p.add_argument('--domain', type=Path, required=True)
p.add_argument('--observer', type=Path, required=True)
p.add_argument('--share', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--test-hooks', action='store_true')
a = p.parse_args()
for name in ('platform', 'host', 'domain', 'observer', 'share', 'out'):
    setattr(a, name, getattr(a, name).resolve())
a.out.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, PYTHONPATH=str(a.host), CAPSTONE_QEMU_LOCK=str(a.out / 'qemu.lock'))
base = [sys.executable, '-m', 'capstone_vm', '--state', str(a.out / 'vm')]
rows = []


def vm(name, *args, timeout=120):
    r = subprocess.run(base + list(args), env=env, capture_output=True, text=True, timeout=timeout)
    (a.out / (name + '.out')).write_text(r.stdout)
    (a.out / (name + '.err')).write_text(r.stderr)
    return r


def check(name, r, code, marker):
    text = r.stdout + r.stderr
    ok = r.returncode == code and marker in text and 'Assertion failed:' not in text
    rows.append(dict(case=name, rc=r.returncode, expected_rc=code, marker=marker, passed=ok))
    print(f'{name}: rc={r.returncode} {"PASS" if ok else "FAIL"}', flush=True)


try:
    pl = a.platform
    r = vm('up', 'up', '--profile', 'virtual', '--memory', '8G', '--share-model', 'mapped-xattr',
           '--share', str(a.share), '--exact-bounds', '--cpu-prop', 'x-capstone-test-csrs=true',
           '--cpu-prop', 'x-capstone-supervisor-slots=2', '--qemu', str(pl / 'qemu-system-riscv64'),
           '--kernel', str(pl / 'images/Image'), '--firmware', str(pl / 'images/fw_jump.elf'),
           '--rootfs', str(pl / 'images/rootfs.ext2'), '--launcher', str(pl / 'adapter/capstone-vexec'),
           '--job-helper', str(pl / 'adapter/capstone-job'), '--module', str(pl / 'adapter/module/capstone_vm.ko'),
           '--ssh-server', str(pl / 'ssh-server/dropbearmulti'))
    if r.returncode:
        raise RuntimeError('guest boot failed')
    paths = [str(Path('/mnt/host') / f.relative_to(a.share)) for f in (a.domain, a.observer)]
    r = vm('stage', 'exec', '--', 'sh', '-c',
           f'cp {shlex.quote(paths[0])} /tmp/iov1.dom && cp {shlex.quote(paths[1])} /tmp/iov1-observe && '
           'cp /mnt/control/launcher /tmp/iov1-launcher && chmod 755 /tmp/iov1.dom /tmp/iov1-observe /tmp/iov1-launcher && echo 0 > /proc/sys/kernel/randomize_va_space')
    if r.returncode:
        raise RuntimeError('binary staging failed')
    params = 'supervisor_slots=2' + (' inline_test_transport=1' if a.test_hooks else '')
    r = vm('module', 'exec', '--', 'sh', '-c', 'rmmod capstone_vm && insmod /mnt/control/module ' + params)
    if r.returncode:
        raise RuntimeError('module reload failed')
    for case in ('functional', 'signal', 'restart', 'heap-libc', 'stale-libc'):
        r = vm(case, 'run', '--cwd', '/tmp', '--', '/tmp/iov1.dom', case)
        check(case, r, 139 if 'libc' in case else 0, 'cause=25' if 'libc' in case else 'PASS')
    for case in ('span', 'outside', 'text', 'retired') + (() if a.test_hooks else ('forged',)):
        r = vm(case, 'exec', '--', '/tmp/iov1-observe', '/tmp/iov1-launcher', '/tmp/iov1.dom', case)
        check(case, r, 0, 'pipe_bytes=256 PASS')
    if a.test_hooks:
        for hold in (1, 0):
            r = vm(f'hold{hold}', 'exec', '--', 'sh', '-c',
                   f'echo {hold} > /sys/module/capstone_vm/parameters/inline_hold')
            if r.returncode:
                raise RuntimeError('test hold control failed')
            r = vm(f'recycle{hold}', 'run', '--cwd', '/tmp', '--', '/tmp/iov1.dom', 'recycle')
            check(f'recycle{hold}', r, 0, 'changed=0 PASS' if hold else 'changed=256 LANDED')
    vm('parameters', 'exec', '--', 'sh', '-c',
       'for p in /sys/module/capstone_vm/parameters/inline*; do echo "$(basename "$p")=$(cat "$p")"; done')
except (RuntimeError, subprocess.TimeoutExpired) as exc:
    rows.append(dict(case='harness', passed=False, error=str(exc)))
finally:
    vm('down', 'down', timeout=30)
    (a.out / 'result.json').write_text(json.dumps(rows, indent=2) + '\n')
sys.exit(0 if rows and all(r['passed'] for r in rows) else 1)
