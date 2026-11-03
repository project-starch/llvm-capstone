#!/usr/bin/env python3
"""Compare table and constructor initialization, with an omitted-store control."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CHECKS = {'before_constructor', 'data', 'interior', 'one_past', 'weak_data_alias',
          'null', 'function', 'weak_function_alias', 'call', 'labels'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('sdk', 'platform', 'work'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    modes = {'table-o0': ['-O0'], 'table-o2': ['-O2'],
             'constructor': ['-O2', '-mllvm', '-capstone-cap-init-table=false'],
             'missing-store': ['-O2', '-mllvm', '-capstone-cap-init-limit=1']}
    for name, flags in modes.items():
        subprocess.run([str(a.sdk / 'capstone-cc'), *flags,
                        str(HERE / 'cap-init-contract.c'), '-o',
                        str(stage / (name + '.dom'))], check=True)
    for src, dest in [('adapter/capstone-vexec', 'capstone-vexec'),
                      ('adapter/module/capstone_vm.ko', 'capstone_vm.ko')]:
        shutil.copy2(a.platform / src, stage / dest)
    commands = ['#!/bin/sh', 'cd /mnt/vm || exit 1', 'dmesg -n 1',
                'insmod capstone_vm.ko || exit 1']
    for name in modes:
        commands += [f'echo CAP_INIT_ARM:{name}',
                     f'./capstone-vexec {name}.dom',
                     f'echo CAP_INIT_EXIT:{name}:$?']
    commands += ['rmmod capstone_vm', 'echo CAP_INIT_UNLOAD:$?',
                 'echo VIRTUAL_STAGED_DONE']
    (stage / 'gate.sh').write_text('\n'.join(commands) + '\n')
    subprocess.run([sys.executable, str(HERE / 'run-staged.py'),
                    '--qemu', str((a.platform / 'qemu-system-riscv64').resolve()),
                    '--images', str((a.platform / 'images').resolve()),
                    '--stage', str(stage.resolve()),
                    '--work', str((a.work / 'guest').resolve()), '--timeout', '180'],
                   check=True)
    log = (a.work / 'guest/serial.log').read_text(errors='replace').replace('\r', '')
    found, arm = {name: {} for name in modes}, None
    for line in log.splitlines():
        if m := re.fullmatch(r'CAP_INIT_ARM:(\S+)', line):
            arm = m[1]
        elif (m := re.fullmatch(r'CAP_INIT_CHECK:(\w+):(PASS|FAIL)', line)) and arm:
            found[arm][m[1]] = m[2]
    exits = dict(re.findall(r'CAP_INIT_EXIT:([^:]+):(\d+)', log))
    tests = {name: set(found[name]) == CHECKS and
                   set(found[name].values()) == {'PASS'} and exits.get(name) == '0'
             for name in modes if name != 'missing-store'}
    tests['missing_store_control'] = (exits.get('missing-store') == '1' and
                                      'FAIL' in found['missing-store'].values())
    tests['unload'] = 'CAP_INIT_UNLOAD:0' in log
    inputs = [HERE / 'cap-init-contract.c', HERE / 'start.S',
              a.sdk / 'sdk.json', *sorted(stage.glob('*.dom'))]
    result = {'status': 'PASS' if all(tests.values()) else 'FAIL',
              'tests': tests, 'checks': found, 'exits': exits,
              'sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in inputs}}
    (a.work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'tests', 'exits')}))
    return 0 if all(tests.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
