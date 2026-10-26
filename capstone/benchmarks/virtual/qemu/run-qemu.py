#!/usr/bin/env python3
"""Functional pass of the microbenchmark images on the virtual-profile QEMU kit, many images per boot.

  run-qemu.py --kit KIT --work DIR --set T1=<build dir> --set T2=<build dir> --native <build dir>
              [--set T1c=<count build dir> ...] [--args "1 5"] [--icount] [--movc-null-scalar]

Each --set is a build.py output directory (its images.tsv names the images). One boot runs every
image of every set, in order, each under the walltime helper (a per-image timeout, and the whole
run's guest time). Results go to DIR/results.json; the program prints one line per set and image.
This is a CORRECTNESS pass: the guest clock under QEMU is not a performance measurement, and with
--icount it is an instruction count, used only to size iterations. The boot is driven through
capstone/runtime/virtual/run-staged.py, which takes the shared QEMU lock (CAPSTONE_QEMU_LOCK).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUN_STAGED = REPO / 'capstone' / 'runtime' / 'virtual' / 'run-staged.py'


def load_set(spec):
    label, _, path = spec.partition('=')
    path = Path(path).resolve()
    names = [l.split('\t')[0] for l in (path / 'images.tsv').read_text().splitlines()[1:] if l.strip()]
    return label, path, names


def parse(log, sets):
    """Per (label, name): the BENCH/ALLOC lines, the exit status and the guest time."""
    res = {}
    cur = None
    for raw in log.splitlines():
        line = raw.rstrip('\r')
        m = re.match(r'RUN-BEGIN (\S+) (\S+)$', line)
        if m:
            cur = res.setdefault(m.group(1), {}).setdefault(m.group(2), {
                'bench': [], 'alloc': [], 'info': None, 'done': None, 'wall': None, 'rc': None,
                'timeout': False, 'other': []})
            continue
        if cur is None:
            continue
        if re.match(r'RUN-END ', line):
            cur = None
            continue
        if m := re.match(r'BENCH (\S+) it=(\S+) cycles=(\d+) instret=(\d+) entries=(\d+) ns=(-?\d+) check=(0x[0-9a-f]+)( VERIFY-FAILED)?$', line):
            cur['bench'].append(dict(id=m.group(1), it=m.group(2), cycles=int(m.group(3)),
                                     instret=int(m.group(4)), entries=int(m.group(5)), ns=int(m.group(6)),
                                     check=m.group(7), ok=m.group(8) is None))
        elif line.startswith('BENCH ') and not line.startswith('BENCH-'):
            cur['other'].append('UNPARSED-BENCH ' + line[:300])
        elif m := re.match(r'ALLOC (\S+) it=(\S+) malloc=(\d+) free=(\d+) realloc=(\d+) calloc=(\d+)$', line):
            cur['alloc'].append(dict(it=m.group(2), malloc=int(m.group(3)), free=int(m.group(4)),
                                     realloc=int(m.group(5)), calloc=int(m.group(6))))
        elif m := re.match(r'BENCH-INFO (\S+) counters=(\S+) warmup=(\d+) iterations=(\d+)$', line):
            cur['info'] = dict(id=m.group(1), counters=m.group(2), warmup=int(m.group(3)), iterations=int(m.group(4)))
        elif m := re.match(r'BENCH-INFO ctl/loop predicted_instret_minus_empty=(\d+)', line):
            cur['predicted_instret_minus_empty'] = int(m.group(1))
        elif m := re.match(r'BENCH-DONE (\S+) failed=(\d+)$', line):
            cur['done'] = int(m.group(2))
        elif m := re.match(r'WALL ns=(\d+) rc=(\d+)( timeout)?$', line):
            cur['wall'], cur['rc'], cur['timeout'] = int(m.group(1)), int(m.group(2)), bool(m.group(3))
        else:
            if line.strip() and len(cur['other']) < 60:
                cur['other'].append(line[:300])
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--kit', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--set', action='append', default=[], help='LABEL=build dir (capstone images)')
    p.add_argument('--native', type=Path, help='build dir of the native references (label "native")')
    p.add_argument('--walltime', type=Path, required=True, help='the static walltime helper')
    p.add_argument('--args', default='1 5', help='W I passed to every image')
    p.add_argument('--only', help='comma-separated names')
    p.add_argument('--nonreentrant', type=Path,
                   help='file of program names run with "0 1" (one timed iteration per process) instead of --args')
    p.add_argument('--image-timeout', type=int, default=300)
    p.add_argument('--timeout', type=int, default=7200, help='whole boot')
    p.add_argument('--icount', action='store_true', help='-icount shift=0,sleep=off (sizing only)')
    p.add_argument('--movc-null-scalar', action='store_true', help='CAPSTONE_MOVC_NULL_SCALAR=1 for QEMU')
    p.add_argument('--extra', action='append', default=[], help='LABEL=path of one extra image to run (a control)')
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    kit = a.kit.resolve()
    shutil.copy2(kit / 'adapter' / 'capstone-vexec', stage / 'capstone-vexec')
    shutil.copy2(kit / 'adapter' / 'module' / 'capstone_vm.ko', stage / 'capstone_vm.ko')
    shutil.copy2(a.walltime, stage / 'walltime')
    only = set(a.only.split(',')) if a.only else None
    nonre = set()
    if a.nonreentrant:
        nonre = {l.split()[0] for l in a.nonreentrant.read_text().splitlines() if l.strip() and not l.startswith('#')}
    sets = [load_set(s) for s in a.set]
    if a.native:
        sets.append(('native',) + load_set('native=' + str(a.native))[1:])
    lines = ['#!/bin/sh', 'cd /mnt/vm || exit 1', 'dmesg -n 1', 'insmod capstone_vm.ko || { echo INSMOD-FAILED; exit 1; }',
             'echo GATE-START', 'dmesg -c > /dev/null']
    plan = []
    for label, path, names in sets:
        (stage / label).mkdir()
        for n in names:
            if only and n not in only:
                continue
            native = label == 'native'
            src = path / (n if native else n + '.dom')
            dst = stage / label / src.name
            shutil.copy2(src, dst)
            cmd = f'./{label}/{src.name}' if native else f'./capstone-vexec {label}/{src.name}'
            plan.append((label, n))
            lines += [f'echo "RUN-BEGIN {label} {n}"',
                      f'./walltime {a.image_timeout} {cmd} {"0 1" if re.sub(r"[.]K[0-9]+$", "", n) in nonre else a.args}',
                      'echo "DMESG-BEGIN"; dmesg -c | grep -v "remote fence extension" | tail -n 20; echo "DMESG-END"',
                      f'echo "RUN-END {label} {n}"']
    for spec in a.extra:
        label, _, path = spec.partition('=')
        (stage / 'extra').mkdir(exist_ok=True)
        shutil.copy2(path, stage / 'extra' / Path(path).name)
        plan.append(('extra', label))
        lines += [f'echo "RUN-BEGIN extra {label}"',
                  f'./walltime {a.image_timeout} ./capstone-vexec extra/{Path(path).name} {a.args}',
                  'echo "DMESG-BEGIN"; dmesg -c | grep -v "remote fence extension" | tail -n 20; echo "DMESG-END"', f'echo "RUN-END extra {label}"']
    lines += ['rmmod capstone_vm; echo "RMMOD:$?"', 'echo VIRTUAL_STAGED_DONE']
    gate = stage / 'gate.sh'
    gate.write_text('\n'.join(lines) + '\n')
    gate.chmod(0o755)
    size = sum(f.stat().st_size for f in stage.rglob('*') if f.is_file())
    disk_mib = max(256, (size >> 20) * 2 + 128)
    qemu = kit / 'qemu-system-riscv64'
    if a.icount:
        wrapper = a.work / 'qemu-icount.sh'
        wrapper.write_text(f'#!/bin/sh\nexec "{qemu}" "$@" -icount shift=0,sleep=off\n')
        wrapper.chmod(0o755)
        qemu = wrapper
    env = dict(os.environ)
    env.pop('CAPSTONE_QEMU_DOMAIN_LOCK', None)
    if a.movc_null_scalar:
        env['CAPSTONE_MOVC_NULL_SCALAR'] = '1'
    else:
        env.pop('CAPSTONE_MOVC_NULL_SCALAR', None)
    meta = dict(kit=str(kit), args=a.args, icount=a.icount, movc_null_scalar=a.movc_null_scalar,
                sets={l: str(p) for l, p, _ in sets}, plan=plan)
    (a.work / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    r = subprocess.run([sys.executable, str(RUN_STAGED), '--qemu', str(qemu), '--images', str(kit / 'images'),
                        '--stage', str(stage), '--work', str(a.work / 'guest'), '--timeout', str(a.timeout),
                        '--disk-mib', str(disk_mib)], env=env)
    serial = a.work / 'guest' / 'serial.log'
    log = serial.read_text(errors='replace') if serial.exists() else ''
    res = parse(log, sets)
    missing = [f'{l}/{n}' for l, n in plan if n not in res.get(l, {})]
    out = dict(meta=meta, completed=r.returncode == 0, missing=missing, results=res,
               gate_started='GATE-START' in log, rmmod='RMMOD:0' in log)
    (a.work / 'results.json').write_text(json.dumps(out, indent=1) + '\n')
    for l, n in plan:
        e = res.get(l, {}).get(n)
        if not e:
            print(f'{l:8s} {n:28s} NO-RESULT')
            continue
        oks = sum(b['ok'] for b in e['bench'])
        checks = sorted({b['check'] for b in e['bench']})
        print(f'{l:8s} {n:28s} rc={e["rc"]} iters={len(e["bench"])} ok={oks} checks={",".join(checks)[:60]} wall={e["wall"]}')
    print(f'completed={out["completed"]} missing={len(missing)}')
    return 0 if out['completed'] and not missing else 1


if __name__ == '__main__':
    sys.exit(main())
