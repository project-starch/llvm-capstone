#!/usr/bin/env python3
"""Run images on virtual Capstone in ONE staged boot (runtime/virtual/run-staged.py), each in an empty
directory, with its own arguments.

    run-reuse-capstone.py --platform DIR --work NEW --run NAME=IMAGE[:ARGS] [--run ...] [--timeout S]
                          [--per-process S]

platform: qemu-system-riscv64, images/{fw_jump.elf,Image,rootfs.ext2}, adapter/{capstone-vexec,
module/capstone_vm.ko}, as the dev gates use it. Each process's stdout and stderr go to files (never a
pipe: a pipe changed PoisonCap's speedtest1 at start-up), which are printed after it ends, between
@@BEGIN/@@END markers with its exit status and the launcher's fault record (CAPSTONE_FAULT_RECORD).
WORK/results.jsonl, one row per process: name, image sha256, arguments, exit status, the fault record,
stdout, stderr. A process that did not run, or a boot that did not finish, fails the run (exit 1).
CAPSTONE_QEMU_CPU_EXTRA and CAPSTONE_QEMU_LOCK pass through to run-staged.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[5]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--platform', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--run', action='append', required=True, metavar='NAME=IMAGE[:ARGS]')
    p.add_argument('--timeout', type=int, default=3600, help='the whole boot')
    p.add_argument('--module-args', default='',
                   help='capstone_vm module parameters at insmod, e.g. node_flat_records=16777216 (the flat '
                        "lifetime table's default, 65,536 records, is exhausted from speedtest1 --size 20 on: "
                        'cause 30); recorded in WORK/results.jsonl')
    p.add_argument('--per-process', type=int, default=0,
                   help='seconds per process, then the process is killed (exit 143); 0: none. A watchdog in the '
                        'gate script: the guest rootfs has no timeout(1)')
    a = p.parse_args()
    runs = []
    for spec in a.run:
        name, _, rest = spec.partition('=')
        image, _, args = rest.partition(':')
        runs.append((name, Path(image), args))
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    shutil.copy2(a.platform / 'adapter/capstone-vexec', stage / 'capstone-vexec')
    shutil.copy2(a.platform / 'adapter/module/capstone_vm.ko', stage / 'capstone_vm.ko')
    lines = ['#!/bin/sh', 'dmesg -n 1',
             f'insmod /mnt/vm/capstone_vm.ko {a.module_args} || {{ echo "@@INSMOD failed"; echo VIRTUAL_STAGED_DONE; exit 1; }}',
             'echo "@@PARAMS $(cat /sys/module/capstone_vm/parameters/node_flat_records)"',
             'cp /mnt/vm/capstone-vexec /tmp/vexec && chmod 755 /tmp/vexec']
    for i, (name, image, args) in enumerate(runs):
        shutil.copy2(image, stage / f'i{i}.dom')
        d = f'/tmp/w{i}'
        lines += [f'rm -rf {d}; mkdir -p {d} && cd {d} && cp /mnt/vm/i{i}.dom x.dom && chmod 755 x.dom',
                  f'echo "@@BEGIN {name}"',
                  f'CAPSTONE_FAULT_RECORD={d}/fault /tmp/vexec {d}/x.dom {args} > {d}/out 2> {d}/err & pid=$!',
                  (f'( sleep {a.per_process}; kill $pid 2>/dev/null ) & dog=$!' if a.per_process else 'dog='),
                  'wait $pid; rc=$?; [ -n "$dog" ] && kill $dog 2>/dev/null',
                  f'echo "@@OUT {name}"; cat {d}/out; echo "@@ERR {name}"; cat {d}/err',
                  f'echo "@@FAULT {name}"; [ -f {d}/fault ] && cat {d}/fault',
                  f'echo "@@END {name} rc=$rc"', 'cd /']
    lines += ['rmmod capstone_vm', 'echo VIRTUAL_STAGED_DONE']
    (stage / 'gate.sh').write_text('\n'.join(lines) + '\n')
    (stage / 'gate.sh').chmod(0o755)
    done = subprocess.run([sys.executable, str(REPO / 'capstone/runtime/virtual/run-staged.py'),
                           '--qemu', str(a.platform / 'qemu-system-riscv64'), '--images', str(a.platform / 'images'),
                           '--stage', str(stage), '--work', str(a.work / 'boot'), '--timeout', str(a.timeout),
                           '--disk-mib', '512'], capture_output=True, text=True)
    (a.work / 'run-staged.out').write_text(done.stdout + done.stderr)
    serial = (a.work / 'boot/serial.log').read_bytes().decode(errors='replace').replace('\r', '')
    failures = [] if '\nVIRTUAL_STAGED_DONE\n' in serial else ['the boot did not finish']
    if '@@INSMOD failed' in serial:
        failures.append(f'insmod failed with {a.module_args!r}')
    fm = re.search(r'^@@PARAMS (\d+)$', serial, re.M)
    flat = int(fm.group(1)) if fm else None
    with (a.work / 'results.jsonl').open('w') as out:
        for name, image, args in runs:
            m = re.search(rf'^@@BEGIN {re.escape(name)}\n(.*?)^@@END {re.escape(name)} rc=(\d+)$', serial, re.M | re.S)
            row = {'name': name, 'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(), 'args': args,
                   'module_args': a.module_args, 'node_flat_records': flat, 'ran': bool(m)}
            if m:
                body = m.group(1)
                part = lambda tag, nxt: (re.search(rf'^@@{tag} {re.escape(name)}\n(.*?)^@@{nxt} ', body + f'@@END {name} ',
                                                   re.M | re.S) or [None, ''])[1]
                row.update(rc=int(m.group(2)), stdout=part('OUT', 'ERR'), stderr=part('ERR', 'FAULT'),
                           fault=part('FAULT', 'END').strip())
            else:
                failures.append(f'{name}: did not run')
            out.write(json.dumps(row) + '\n')
            print(f"  {name:<34} ran={row['ran']} rc={row.get('rc')} fault={'yes' if row.get('fault') else 'no'}")
    for f in failures:
        print('FAILED', f)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
