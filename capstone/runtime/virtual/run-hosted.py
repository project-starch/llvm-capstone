#!/usr/bin/env python3
"""Qualify hosted C semantics and tag-preserving copies in virtual applications.

Runs the native control, then hosted-contract.c and hosted-noreturn.c as
virtual images in one boot, and records every check with input hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('adapter', 'application', 'noreturn', 'native', 'qemu', 'images', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    inputs = dict(launcher=a.adapter / 'capstone-vexec', module=a.adapter / 'module/capstone_vm.ko',
                  application=a.application, noreturn=a.noreturn, native=a.native, qemu=a.qemu,
                  source=HERE / 'hosted-contract.c', noreturn_source=HERE / 'hosted-noreturn.c')
    inputs.update({name: a.images / name for name in ('Image', 'fw_jump.elf', 'rootfs.ext2')})
    for key, name in [('launcher', 'capstone-vexec'), ('module', 'capstone_vm.ko'),
                      ('application', 'hosted.dom'), ('noreturn', 'noreturn.dom'),
                      ('native', 'native-hosted')]:
        shutil.copy2(inputs[key], stage / name)
    (stage / 'gate.sh').write_text('''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
echo HOSTED_ARM:native
./native-hosted
echo HOSTED_EXIT:native:$?
insmod capstone_vm.ko || exit 1
echo HOSTED_ARM:virtual
./capstone-vexec hosted.dom
echo HOSTED_EXIT:virtual:$?
./capstone-vexec noreturn.dom
echo HOSTED_EXIT:noreturn:$?
rmmod capstone_vm
echo HOSTED_CLEANUP:$?
echo VIRTUAL_STAGED_DONE
''')
    subprocess.run([sys.executable, str(HERE / 'run-staged.py'), '--qemu', str(a.qemu.resolve()),
                    '--images', str(a.images.resolve()), '--stage', str(stage.resolve()),
                    '--work', str((a.work / 'guest').resolve()), '--timeout', str(a.timeout),
                    '--exact-bounds'])
    serial = a.work / 'guest/serial.log'
    log = serial.read_text(errors='replace').replace('\r', '') if serial.exists() else ''
    arms, arm = {'native': {}, 'virtual': {}}, None
    for line in log.splitlines():
        if m := re.match(r'HOSTED_ARM:(\w+)$', line):
            arm = m.group(1)
        elif (m := re.match(r'HOSTED_CHECK:(\w+):(PASS|FAIL)$', line)) and arm:
            arms[arm][m.group(1)] = m.group(2) == 'PASS'
    exits = dict(re.findall(r'HOSTED_EXIT:(\w+):(\d+)', log))
    tests = {f'{arm}:{name}': ok for arm, found in arms.items() for name, ok in found.items()}
    tests['native:exit'] = exits.get('native') == '0'
    tests['virtual:exit'] = exits.get('virtual') == '0'
    tests['noreturn:exit_zero'] = exits.get('noreturn') == '0'
    tests['noreturn:ran'] = 'HOSTED_NORETURN:41' in log
    tests['module_unloaded'] = 'HOSTED_CLEANUP:0' in log
    tests['virtual_tag_control_fired'] = arms['virtual'].get('control_byte_copy_drops_tag') is True
    tests['virtual_pipe_control_fired'] = arms['virtual'].get('control_pipe_read_returns_available') is True
    # The frame- and return-address checks, by name: an image built before them
    # would otherwise pass on the remaining checks.
    portable = {'return_address_returned', 'return_address_argument',
                'return_address_call_site_order'}
    tests['virtual_frame_checks_parsed'] = portable | {
        'return_address_untagged', 'return_address_depth_1',
        'frame_address_depth_1'} <= set(arms['virtual'])
    tests['native_frame_checks_parsed'] = portable <= set(arms['native'])
    # No checks parsed is a failed run, never an empty pass.
    tests['virtual_checks_parsed'] = len(arms['virtual']) >= 21
    tests['native_checks_parsed'] = len(arms['native']) >= 18
    result = dict(status='PASS' if all(tests.values()) else 'FAIL',
                  passed=sum(tests.values()), total=len(tests), tests=tests,
                  exits=exits, sha256={k: sha(v) for k, v in inputs.items()})
    (a.work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'passed', 'total', 'exits')}))
    for name, ok in tests.items():
        if not ok:
            print('FAIL', name)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
