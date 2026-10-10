#!/usr/bin/env python3
"""Qualify local Capstone mallocng, its VM slow paths and object lifetimes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import struct
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('adapter', 'application', 'native', 'qemu', 'images', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--skip-churn', action='store_true')
    a = p.parse_args()
    nm = Path(os.environ.get('CAPSTONE_LLVM_BIN', '')) / 'llvm-nm'
    symbols = {name: int(value, 16) for value, _, name in
               (line.split() for line in subprocess.check_output([str(nm), str(a.application)],
                                                                text=True).splitlines() if len(line.split()) == 3)}
    required = {'cap_malloc_fault_store', 'cap_malloc_validate', 'cap_malloc_invalid_free'}
    if not required <= symbols.keys():
        p.error('application lacks current safety sites; rebuild its SDK and relink')
    elf = a.application.read_bytes()
    phoff = struct.unpack_from('<Q', elf, 32)[0]
    phsize, phnum = struct.unpack_from('<HH', elf, 54)
    image_base = min(struct.unpack_from('<Q', elf, phoff + i * phsize + 16)[0]
                     for i in range(phnum) if struct.unpack_from('<I', elf, phoff + i * phsize)[0] == 1)
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    inputs = dict(launcher=a.adapter / 'capstone-vexec', module=a.adapter / 'module/capstone_vm.ko',
                  application=a.application, native=a.native, qemu=a.qemu,
                  source=HERE / 'malloc-contract.c')
    inputs.update({name: a.images / name for name in ('Image', 'fw_jump.elf', 'rootfs.ext2')})
    for key, name in [('launcher', 'capstone-vexec'), ('module', 'capstone_vm.ko'),
                      ('application', 'malloc.dom'), ('native', 'native-malloc')]:
        shutil.copy2(inputs[key], stage / name)
    gate = '''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
./native-malloc ok
echo MALLOC_NATIVE_EXIT:$?
insmod capstone_vm.ko || exit 1
CAPSTONE_VM_STATS=1 CAPSTONE_VM_TRACE=1 ./capstone-vexec malloc.dom ok
echo MALLOC_BASIC_EXIT:$?
CAPSTONE_VM_STATS=1 ./capstone-vexec malloc.dom population
echo MALLOC_POPULATION_EXIT:$?
'''
    if not a.skip_churn:
        gate += 'CAPSTONE_VM_STATS=1 ./capstone-vexec malloc.dom churn\necho MALLOC_CHURN_EXIT:$?\n'
    gate += '''CAPSTONE_VM_TRACE=1 ./capstone-vexec malloc.dom hot
echo MALLOC_HOT_EXIT:$?
CAPSTONE_VM_TRACE=1 ./capstone-vexec malloc.dom hot-service-control
echo MALLOC_HOT_CONTROL_EXIT:$?
for case in stale inplace-stale remap-stale bounds stale-free reuse-stale reuse-free interior-free; do
  ./capstone-vexec malloc.dom "$case"
  echo MALLOC_DENIAL_EXIT:$case:$?
done
rmmod capstone_vm
echo MALLOC_CLEANUP:$?
echo VIRTUAL_STAGED_DONE
'''
    (stage / 'gate.sh').write_text(gate)
    run = subprocess.run([sys.executable, str(HERE / 'run-staged.py'), '--qemu', str(a.qemu.resolve()),
                          '--images', str(a.images.resolve()), '--stage', str(stage.resolve()),
                          '--work', str((a.work / 'guest').resolve()), '--timeout', str(a.timeout),
                          '--exact-bounds'])
    log = (a.work / 'guest/serial.log').read_text(errors='replace').replace('\r', '')
    checks = {
        'native_reference': 'MALLOC_NATIVE_EXIT:0\n' in log,
        'basic_inplace_mremap_failure': 'MALLOC_BASIC_EXIT:0\n' in log,
        'nonlinear_and_linear_tag_transfer': log.count('MALLOC_TAG_COPY_OK\n') == 2,
        '70000_simultaneously_live_objects': 'MALLOC_POPULATION_OK\n' in log and 'MALLOC_POPULATION_EXIT:0\n' in log,
        'cleanup': run.returncode == 0 and 'MALLOC_CLEANUP:0\n' in log,
        'no_kernel_fault': 'Kernel panic' not in log and 'Oops' not in log,
    }
    if not a.skip_churn:
        checks['200000_lifetimes'] = 'MALLOC_CHURN_OK\n' in log and 'MALLOC_CHURN_EXIT:0\n' in log
        checks['collection_exercised'] = any(int(n) > 0 for n in re.findall(r'collections=(\d+)', log))
    # Match only after the setup marker, up to this process's exit marker.
    sites = [('reuse-stale', 24, 'cap_malloc_fault_store'), ('reuse-free', 24, 'cap_malloc_validate'),
             ('stale', 24, 'cap_malloc_fault_store'), ('remap-stale', 24, 'cap_malloc_fault_store'), ('inplace-stale', 24, 'cap_malloc_fault_store'),
             ('bounds', 28, 'cap_malloc_fault_store'), ('stale-free', 24, 'cap_malloc_validate'),
             ('interior-free', 2, 'cap_malloc_invalid_free')]
    for name, cause, symbol in sites:
        match = re.search(r'MALLOC_FAULT_READY:' + re.escape(name) + r'\n(.*?)MALLOC_DENIAL_EXIT:' +
                          re.escape(name) + r':139\n', log, re.S)
        fault = re.search(r'domain fault cause=(\d+) pc=0x([0-9a-f]+).*?code=0x([0-9a-f]+)-',
                          match[1]) if match else None
        checks[name] = bool(fault and int(fault[1]) == cause and
                            int(fault[2], 16) - int(fault[3], 16) == symbols[symbol] - image_base)
    hot = re.search(r'MALLOC_HOT_BEGIN\n(.*?)MALLOC_HOT_END\n', log, re.S)
    kinds = re.findall(r'CAPSTONE_VM_BOUNDARY kind=(\d+)', hot[1]) if hot else []
    # The end marker asks for the thread identity (11) and delegates its write
    # through the thread transport (10). No allocator service is permitted.
    checks['local_allocation_fast_path'] = bool(hot and kinds and kinds == ['11', '10'] and
                                               'MALLOC_HOT_EXIT:0\n' in log)
    controls = list(re.finditer(r'MALLOC_HOT_BEGIN\n(.*?)MALLOC_HOT_END\n', log, re.S))
    control_kinds = re.findall(r'CAPSTONE_VM_BOUNDARY kind=(\d+)', controls[1][1]) if len(controls) == 2 else []
    checks['fast_path_control_detects_service'] = ('7' in control_kinds and control_kinds != ['11', '10'] and
                                                'MALLOC_HOT_CONTROL_EXIT:0\n' in log)
    checks['real_linux_mremap_growth_and_shrink'] = log.count('CAPSTONE_VM_REMAP_OK') >= 2
    record = dict(status='PASS' if all(checks.values()) else 'FAIL', tests=checks,
                  virtual_vm_abi=5, harts=1, allocator='Capstone-compiled musl 1.2.5 mallocng; upstream allocation policy',
                  exact_bounds='physical shadow metadata; QEMU prototype', skipped=['churn'] if a.skip_churn else [],
                  input_sha256={name: sha(path) for name, path in inputs.items()},
                  gate_sha256=hashlib.sha256(gate.encode()).hexdigest(), runner_sha256=sha(Path(__file__)))
    (a.work / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
