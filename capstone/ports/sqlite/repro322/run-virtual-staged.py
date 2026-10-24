#!/usr/bin/env python3
"""Run a build-virtual.py build of the SQLite engine corpus in ONE staged boot (run-staged.py).

    run-virtual-staged.py --build DIR --platform DIR --work NEW --llvm-bin DIR [--cases temporal|all|NN,NN]

The persistent-VM runner (bug-corpora/sqlite/engine-repros/shared/run-virtual.py) needs a
capstone_vm state; this one needs only a platform directory as the dev gates use it
(qemu-system-riscv64, images/{fw_jump.elf,Image,rootfs.ext2}, adapter/{capstone-vexec,
module/capstone_vm.ko}). The two controls (control_uaf_mem5, control_bounds_mem5, built with
--observe) run first, then the cases, each as `capstone-vexec IMAGE TAG` in the guest, its output
between @@BEGIN and @@END markers on the console. The launcher writes its fault line only to a
terminal or to the file CAPSTONE_FAULT_RECORD names (runtime/linux/exec.c), so each process gets
a record file, printed after its output.

Per process, WORK/results.jsonl: the exit status, BEGIN/RETURNED, and the launcher's domain fault line
read with bug-corpora/tools/verdicts.py (cause, pc, and the function from the image's own symbols),
"at_site" when that function is one of the case's fault_sites, and the process's whole output. The
run exits 1 when anything is no reading: a requested case not built, a process that did not run or
begin, a nonzero exit without a fault record, a zero exit without RETURNED, a fault in no function of
the image, a control that did not read its expected cause at its function, or a boot that did not
finish. --parse-only reads an existing WORK again without booting. CAPSTONE_QEMU_CPU_EXTRA and
CAPSTONE_QEMU_LOCK are passed through to run-staged.py.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
CORPUS = REPO / 'capstone/bug-corpora/sqlite/engine-repros'
sys.path.insert(0, str(REPO / 'capstone/bug-corpora/tools'))
import verdicts as v  # noqa: E402

CONTROLS = (('control_uaf_mem5', 'control_uaf_mem5'), ('control_bounds_mem5', 'control_bounds_mem5'))
# What each control must read: a stale read and an overflowing write, each at its own function.
CONTROL_FAULT = {'control_uaf_mem5': (25, 'control_read'), 'control_bounds_mem5': (28, 'control_write')}


def temporal():
    rows = (CORPUS / 'results/matrix.tsv').read_text().splitlines()[1:]
    return {c[1] for c in (r.split('\t') for r in rows) if len(c) > 2 and c[2] == 'temporal'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--platform', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--llvm-bin', type=v.llvm_bin_dir, required=True)
    p.add_argument('--cases', default='temporal')
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--parse-only', action='store_true',
                   help='Read an existing WORK/boot/serial.log again; build and boot nothing')
    a = p.parse_args()
    build = json.loads((a.build / 'build.json').read_text())
    if a.cases == 'temporal':
        wanted = temporal()
    elif a.cases == 'all':
        wanted = set(build['cases'])
    else:
        wanted = {c for c in build['cases'] if c[:2] in set(a.cases.split(','))}
    missing = sorted(wanted - set(build['built']))
    if missing:
        sys.exit(f'requested but not built: {" ".join(missing)}')
    runs = [(n, t) for n, t in CONTROLS if (a.build / f'{n}.dom').exists()]
    if len(runs) != len(CONTROLS):
        sys.exit('the build lacks its controls (build-virtual.py --observe controls/control_*_mem5.c)')
    runs += [(c, build['cases'][c]['tag']) for c in sorted(wanted)]

    if a.parse_only:
        boot_rc = None
    else:
        a.work.mkdir(parents=True, exist_ok=False)
        stage = a.work / 'stage'
        stage.mkdir()
        shutil.copy2(a.platform / 'adapter/capstone-vexec', stage / 'capstone-vexec')
        shutil.copy2(a.platform / 'adapter/module/capstone_vm.ko', stage / 'capstone_vm.ko')
        lines = ['#!/bin/sh', 'dmesg -n 1',
                 'insmod /mnt/vm/capstone_vm.ko || { echo "@@INSMOD failed"; echo VIRTUAL_STAGED_DONE; exit 1; }',
                 'cp /mnt/vm/capstone-vexec /tmp/vexec && chmod 755 /tmp/vexec', 'cd /tmp']
        for i, (name, tag) in enumerate(runs):
            shutil.copy2(a.build / f'{name}.dom', stage / f'i{i}.dom')
            lines += [f'cp /mnt/vm/i{i}.dom /tmp/x.dom && chmod 755 /tmp/x.dom',
                      f'echo "@@BEGIN {name}"',
                      'rm -f /tmp/f',
                      f'CAPSTONE_FAULT_RECORD=/tmp/f /tmp/vexec /tmp/x.dom {tag} > /tmp/o 2>&1; rc=$?',
                      'cat /tmp/o', '[ -f /tmp/f ] && cat /tmp/f', f'echo "@@END {name} rc=$rc"']
        lines += ['rmmod capstone_vm', 'echo VIRTUAL_STAGED_DONE']
        (stage / 'gate.sh').write_text('\n'.join(lines) + '\n')
        (stage / 'gate.sh').chmod(0o755)
        done = subprocess.run([sys.executable, str(REPO / 'capstone/runtime/virtual/run-staged.py'),
                               '--qemu', str(a.platform / 'qemu-system-riscv64'), '--images', str(a.platform / 'images'),
                               '--stage', str(stage), '--work', str(a.work / 'boot'), '--timeout', str(a.timeout),
                               '--disk-mib', '256'], capture_output=True, text=True)
        (a.work / 'run-staged.out').write_text(done.stdout + done.stderr)
        boot_rc = done.returncode
    serial = (a.work / 'boot/serial.log').read_bytes().decode(errors='replace').replace('\r', '')
    failures = []
    if boot_rc not in (None, 0):
        failures.append(f'run-staged.py exited {boot_rc}')
    if '\nVIRTUAL_STAGED_DONE\n' not in serial:
        failures.append('the gate script never finished (no VIRTUAL_STAGED_DONE)')
    with (a.work / 'results.jsonl').open('w') as out:
        for name, tag in runs:
            m = re.search(rf'^@@BEGIN {re.escape(name)}\n(.*?)^@@END {re.escape(name)} rc=(\d+)', serial, re.M | re.S)
            row = {'name': name, 'tag': tag, 'ran': bool(m)}
            if m:
                text, rc = m.group(1), int(m.group(2))
                fault = v.domain_fault(text, v.Symbols(a.llvm_bin, a.build / f'{name}.dom'))
                sites = set()
                if (CORPUS / name / 'case.json').exists():
                    sites = set(json.loads((CORPUS / name / 'case.json').read_text()).get('fault_sites') or [])
                # The whole process window, without the module's per-slot setup lines, so a reader can
                # see how far a case got: REPRO322_MAIN ignores run_case()'s result, so an exit status
                # of 0 and RETURNED alone do not show that the case did its work.
                output = '\n'.join(l for l in text.splitlines() if not l.startswith('CAPSTONE_VM_ADD_OK'))
                row.update(rc=rc, begin=f'{tag} BEGIN' in text, returned=f'{tag} RETURNED' in text,
                           fault=fault and {'cause': fault.cause, 'pc': hex(fault.pc), 'function': fault.symbol},
                           at_site=bool(fault and fault.symbol in sites), output=output[-20000:])
            out.write(json.dumps(row) + '\n')
            f = row.get('fault')
            print(f"  {name:<52} ran={row['ran']} rc={row.get('rc')} "
                  f"fault={f and f['cause']} in={f and f['function']} at_site={row.get('at_site')}")
            # No reading is a failure, never a clean or uncaught row: a process that did not run, did
            # not begin, died without the launcher's fault record, or faulted at an unresolved pc.
            if not row['ran']:
                failures.append(f'{name}: did not run')
                continue
            if not row['begin']:
                failures.append(f'{name}: no BEGIN line')
            if row['rc'] != 0 and not f:
                failures.append(f'{name}: exit {row["rc"]} without a fault record')
            if row['rc'] == 0 and not row['returned']:
                failures.append(f'{name}: exit 0 without a RETURNED line')
            if f and not f['function']:
                failures.append(f'{name}: fault pc {f["pc"]} in no function of the image')
            want = CONTROL_FAULT.get(name)
            if want and (not f or (f['cause'], f['function']) != want):
                failures.append(f'{name}: expected cause {want[0]} in {want[1]}, read {f and (f["cause"], f["function"])}')
    print('staged rc', boot_rc)
    for failure in failures:
        print('FAILED', failure)
    return 1 if failures else 0

if __name__ == '__main__':
    sys.exit(main())
