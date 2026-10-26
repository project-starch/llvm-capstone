#!/usr/bin/env python3
"""Run the two image shapes of shape-contract.c (no capability globals; a constant capability
table) through the launcher: --application-plain and --application-table are the two builds."""
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
    for name in ('adapter', 'application-plain', 'application-table', 'qemu', 'images', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--exact-bounds', action=argparse.BooleanOptionalAction, default=False)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work/'stage'
    stage.mkdir()
    paths = dict(launcher=a.adapter/'capstone-vexec', module=a.adapter/'module/capstone_vm.ko',
                 plain=a.application_plain, table=a.application_table, qemu=a.qemu, source=HERE/'shape-contract.c')
    paths.update({name: a.images/name for name in ('Image', 'fw_jump.elf', 'rootfs.ext2')})
    for key, name in [('launcher', 'capstone-vexec'), ('module', 'capstone_vm.ko'),
                      ('plain', 'plain.dom'), ('table', 'table.dom')]:
        shutil.copy2(paths[key], stage/name)
    gate = '''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
CAPSTONE_EXEC_DIAGNOSTICS=1 ./capstone-vexec plain.dom
echo PLAIN_EXIT:$?
CAPSTONE_EXEC_DIAGNOSTICS=1 ./capstone-vexec table.dom
echo TABLE_EXIT:$?
rmmod capstone_vm
echo SHAPE_CLEANUP:$?
echo VIRTUAL_STAGED_DONE
'''
    (stage/'gate.sh').write_text(gate)
    run = subprocess.run([sys.executable, str(HERE/'run-staged.py'), '--qemu', str(a.qemu.resolve()),
                          '--images', str(a.images.resolve()), '--stage', str(stage),
                          '--work', str(a.work/'guest'), '--timeout', str(a.timeout)] + (['--exact-bounds'] if a.exact_bounds else ['--no-exact-bounds']))
    log = (a.work/'guest/serial.log').read_text(errors='replace').replace('\r', '')
    checks = {
        'no_capability_globals': log.count('VIRTUAL_SHAPE_OK no_capability_globals') == 1,
        'constant_capability_table': log.count('VIRTUAL_SHAPE_OK constant_capability_table') == 1,
        'no_failure': 'VIRTUAL_SHAPE_FAIL' not in log and log.count('VIRTUAL_SHAPE_DONE failures=0') == 2,
        'plain_exit': log.count('PLAIN_EXIT:0') == 1,
        'table_exit': log.count('TABLE_EXIT:0') == 1,
        'cleanup': run.returncode == 0 and log.count('SHAPE_CLEANUP:0') == 1,
    }
    record = dict(status='PASS' if all(checks.values()) else 'FAIL', tests=checks,
                  input_sha256={name: sha(path) for name, path in paths.items()},
                  gate_sha256=hashlib.sha256(gate.encode()).hexdigest(), runner_sha256=sha(Path(__file__)))
    (a.work/'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
