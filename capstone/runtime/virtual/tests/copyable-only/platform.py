#!/usr/bin/env python3
"""Qualify privileged raw operations and fail-closed Linux platform admission.

Source capstone/tests/capstone-test-env.sh before invoking. No benchmark inputs
or external downloads are needed. Every case boots a fresh snapshot guest.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def raw_faults_valid(serial, symbols):
    """Require exact raw instruction and PC, not merely any startup fault."""
    serial = serial.replace('\r', '')
    for op, instruction in [('L', 0x428b), ('S', 0x500b), ('Q', 0x628b), ('A', 0x10628b)]:
        blocks = re.findall('COPYABLE_RAW_BEGIN:'+op+r'\n(.*?)COPYABLE_RAW_REFUSED:'+op+r'\n',
                            serial, re.S)
        if len(blocks) != 1:
            return False
        faults = re.findall(r'domain fault cause=(\d+) pc=(0x[0-9a-f]+) address=(0x[0-9a-f]+)'
                            r'[^\n]*? entry=(0x[0-9a-f]+)', blocks[0])
        if len(faults) != 1:
            return False
        cause, pc, tval, entry = (int(v, 0) for v in faults[0])
        if (cause != 2 or tval != instruction or
                pc-entry != symbols['raw_site_'+op]-symbols['domain_main']):
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qemu', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--firmware', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dtc', default='dtc', help='Host device-tree compiler for the unsupported-platform fixture')
    p.add_argument('--nm', default='nm', help='ELF symbol reader for exact fault-PC checks')
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    inputs = [a.qemu, a.images/'Image', a.images/'rootfs.ext2', a.firmware, a.probe,
              a.adapter/'capstone-vexec', a.adapter/'module/capstone_vm.ko',
              Path(__file__), here/'platform-gate.sh', here/'raw-op.c',
              here.parents[1]/'run-staged.py']
    symbols = {}
    for line in subprocess.check_output([a.nm, '--defined-only', str(a.probe)], text=True).splitlines():
        fields = line.split()
        if len(fields) == 3:
            symbols[fields[2]] = int(fields[0], 16)
    for name in ['domain_main', *['raw_site_'+op for op in 'LSQA']]:
        if name not in symbols:
            p.error('probe missing symbol: '+name)
    result = {'inputs_sha256': {str(f.resolve()): sha256(f)
                              for f in inputs}, 'cases': []}
    # The real model refuses multiple harts before Linux boots. To test Linux's
    # independent guard, boot scalar RISC-V with a synthetic raw-tag declaration.
    # The kernel must refuse it before executing any raw instruction.
    dtb = a.out/'unsupported.dtb'
    dts = a.out/'unsupported.dts'
    subprocess.run([str(a.qemu), '-M', 'virt-capstone,dumpdtb='+str(dtb),
                    '-cpu', 'rv64,sstc=false,h=false,sv48=false,sv57=false',
                    '-m', '2G', '-smp', '2', '-bios', 'none', '-display', 'none'],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    subprocess.run([a.dtc, '-I', 'dtb', '-O', 'dts', '-o', str(dts), str(dtb)],
                   check=True, stderr=subprocess.PIPE)
    with dts.open('a') as f:
        for name in ['copyable', 'unpinned']:
            f.write('/ { capstone-'+name+' { compatible = "capstone,'+name+'-v1"; '
                    'id-bits = <30>; exact-bounds = <0>; page-size = <4096>; '
                    'granule-size = <16>; }; };\n')
    subprocess.run([a.dtc, '-I', 'dts', '-O', 'dtb', '-o', str(dtb), str(dts)],
                   check=True, stderr=subprocess.PIPE)
    result['inputs_sha256'][str(dtb)] = sha256(dtb)
    cases = [('enabled', 1, True, ['captag.enabled=0'], True),
             ('raw-disabled', 1, False, ['captag.enabled=1'], False),
             ('kernel-disabled', 1, True, ['captag=off', 'captag.enabled=1'], False),
             ('model-two-harts', 2, True, [], False),
             ('kernel-two-harts', 2, True, ['captag.enabled=1'], False)]
    for name, cpus, raw, kernel_args, enabled in cases:
        out = a.out/name
        stage = out/'stage'
        stage.mkdir(parents=True)
        for src in [a.adapter/'capstone-vexec', a.adapter/'module/capstone_vm.ko']:
            shutil.copy2(src, stage/src.name)
        shutil.copy2(a.probe, stage/'raw-op.dom')
        shutil.copy2(here/'platform-gate.sh', stage/'gate.sh')
        (stage/'expected').write_text('enabled\n' if enabled else 'disabled\n')
        env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA=(
            'x-capstone-supervisor-slots=32,x-pcrel=false,x-capstone-test-csrs=false,'
            'x-capstone-raw-moves='+str(raw).lower()))
        cmd = ['python3', str(here.parents[1]/'run-staged.py'),
               '--qemu', str(a.qemu), '--images', str(a.images),
               '--firmware', str(a.firmware), '--stage', str(stage),
               '--work', str(out/'run'), '--cpus', str(cpus),
               '--disk-mib', '64', '--timeout', '120']
        if name == 'kernel-two-harts':
            env['CAPSTONE_QEMU_CPU_EXTRA'] += ',x-capstone-u-mode=false'
            cmd.extend(['--dtb', str(dtb)])
        for arg in kernel_args:
            cmd.extend(['--kernel-arg', arg])
        with (out/'runner.log').open('w') as log:
            rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        serial = (out/'run/serial.log').read_text(errors='replace')
        markers = ['COPYABLE_PLATFORM_EXIT:0', 'VIRTUAL_STAGED_DONE',
                   'COPYABLE_PLATFORM_ENABLED' if enabled else 'COPYABLE_PLATFORM_REFUSED']
        if enabled:
            markers += ['COPYABLE_RAW_REFUSED:'+op for op in 'LSQA']
        passed = (rc == 0 and all(serial.count(m) == 1 for m in markers) and
                  not any(m in serial for m in ['Oops', 'Kernel panic', 'COPYABLE_UNEXPECTED_LOAD']))
        if enabled:
            passed = passed and raw_faults_valid(serial, symbols)
        if name == 'model-two-harts':
            passed = (rc != 0 and 'x-capstone-u-mode requires exactly one possible hart' in serial
                      and 'Linux version' not in serial)
        if name == 'kernel-two-harts':
            passed = passed and 'captag: disabled: requires one possible CPU' in serial
        result['cases'].append({'name': name, 'cpus': cpus, 'raw': raw,
                                'kernel_args': kernel_args, 'pass': passed})
        result['pass'] = len(result['cases']) == len(cases) and all(c['pass'] for c in result['cases'])
        (a.out/'qualification.json').write_text(json.dumps(result, indent=2)+'\n')
        print(('PASS ' if passed else 'FAIL ')+name, flush=True)
        if not passed:
            raise SystemExit('See '+str(out/'run/serial.log'))


if __name__ == '__main__':
    main()
