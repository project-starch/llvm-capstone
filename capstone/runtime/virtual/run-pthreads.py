#!/usr/bin/env python3
"""Qualify musl pthread lifetimes, TLS, synchronization and independent I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('adapter', 'application', 'qemu', 'images', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--exact-bounds', action='store_true')
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work/'stage'
    stage.mkdir()
    paths = dict(launcher=a.adapter/'capstone-vexec', module=a.adapter/'module/capstone_vm.ko',
                 application=a.application, qemu=a.qemu, source=HERE/'pthread-contract.c')
    paths.update({name: a.images/name for name in ('Image', 'fw_jump.elf', 'rootfs.ext2')})
    for key, name in [('launcher', 'capstone-vexec'), ('module', 'capstone_vm.ko'), ('application', 'pthread.dom')]:
        shutil.copy2(paths[key], stage/name)
    gate = '''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
CAPSTONE_EXEC_DIAGNOSTICS=1 ./capstone-vexec pthread.dom
echo PTHREAD_EXIT:$?
rmmod capstone_vm
echo PTHREAD_CLEANUP:$?
echo VIRTUAL_STAGED_DONE
'''
    (stage/'gate.sh').write_text(gate)
    run = subprocess.run([sys.executable, str(HERE/'run-staged.py'), '--qemu', str(a.qemu.resolve()),
                          '--images', str(a.images.resolve()), '--stage', str(stage),
                          '--work', str(a.work/'guest'), '--timeout', str(a.timeout)] +
                         (['--exact-bounds'] if a.exact_bounds else []))
    log = (a.work/'guest/serial.log').read_text(errors='replace').replace('\r', '')
    checks = {name: log.count('VIRTUAL_PTHREAD_OK ' + name) == 1 for name in
              ('unsupported_clone_refused', 'tls mutex cond tagged_join', 'timed_wait', 'independent_blocking_io', 'private_epoll_events', 'shared_tls_transport', '64_joined_lifetimes')}
    checks['exit'] = log.count('PTHREAD_EXIT:0') == 1
    checks['cleanup'] = run.returncode == 0 and log.count('PTHREAD_CLEANUP:0') == 1
    record = dict(status='PASS' if all(checks.values()) else 'FAIL', tests=checks,
                  harts=1, virtual_vm_abi=4 if a.exact_bounds else 3,
                  input_sha256={name: sha(path) for name, path in paths.items()},
                  gate_sha256=hashlib.sha256(gate.encode()).hexdigest(), runner_sha256=sha(Path(__file__)))
    (a.work/'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
