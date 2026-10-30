#!/usr/bin/env python3
"""Generate a perf run script (guest side) and its steps.tsv from a plan: programs in a fixed random order, arms ABBA
with alternating start, processes spread over allocator pads K for the allocating programs.

  make-run.py --images <dir with T1/ T2/> --out <stage dir> --procs P --args "W I" --seed S --programs a,b,c
              [--multik a,b] [--controls ctl-empty,ctl-loop --control-procs 2] [--name pilot]

Each process's STDERR goes to a tmpfs file, never the console: the launcher prints a CAPSTONE_VM_ADD_OK line per admitted
mapping, and on the board the console is a 115,200-baud UART (the first pilot spent minutes per process on it). Only the
line count is printed. Data come from stdout's BENCH lines; the dmesg release line is quoted for cross-checking only.
"""
import argparse, random, os, shutil
ap = argparse.ArgumentParser()
ap.add_argument('--images', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--procs', type=int, required=True); ap.add_argument('--args', default='1 10')
ap.add_argument('--seed', type=int, default=20261026); ap.add_argument('--programs', required=True)
ap.add_argument('--multik', default=''); ap.add_argument('--kmax', type=int, default=4)
ap.add_argument('--controls', default='ctl-empty,ctl-loop'); ap.add_argument('--control-procs', type=int, default=2)
ap.add_argument('--name', default='run')
a = ap.parse_args()
progs = a.programs.split(','); multik = set(filter(None, a.multik.split(',')))
rng = random.Random(a.seed); rng.shuffle(progs)
steps = []
for i, p in enumerate(progs):
    unit = ['T1', 'T2', 'T2', 'T1'] if i % 2 == 0 else ['T2', 'T1', 'T1', 'T2']
    pat = (unit * ((2 * a.procs + 3) // 4))[:2 * a.procs]
    cnt = {'T1': 0, 'T2': 0}
    for arm in pat:
        k = cnt[arm] % a.kmax if p in multik else 0
        steps.append((p, arm, cnt[arm], k)); cnt[arm] += 1
for c in filter(None, a.controls.split(',')):
    unit = ['T1', 'T2', 'T2', 'T1']
    cnt = {'T1': 0, 'T2': 0}
    for arm in (unit * a.control_procs)[:2 * a.control_procs]:
        steps.append((c, arm, cnt[arm], 0)); cnt[arm] += 1
os.makedirs(a.out, exist_ok=True)
for arm in ('T1', 'T2'):
    os.makedirs(f'{a.out}/{arm}', exist_ok=True)
    for (p, ar, _, k) in steps:
        if ar == arm:
            src = f'{a.images}/{arm}/{p}.K{k}.dom'
            if not os.path.exists(src): raise SystemExit(f'missing image {src}')
            shutil.copy(src, f'{a.out}/{arm}/')
with open(f'{a.out}/../steps-{a.name}.tsv', 'w') as f:
    f.write('n\tprogram\tarm\tproc\tK\targs\n')
    for n, s in enumerate(steps): f.write(f'{n}\t{s[0]}\t{s[1]}\t{s[2]}\t{s[3]}\t{a.args}\n')
with open(f'{a.out}/run-{a.name}.sh', 'w') as f:
    f.write(f'#!/bin/sh\n# perf {a.name}: {len(steps)} processes, procs/arm={a.procs}, args "{a.args}", seed {a.seed}. '
            'stderr of every process to tmpfs; console gets stdout and the line count only.\n')
    f.write('cd "$(dirname "$0")" || exit 1\nls /sys/module/capstone_vm/parameters | grep bench\n')
    for n, (p, arm, proc, k) in enumerate(steps):
        f.write(f'echo "@@PERF-BEGIN {n} {p} {arm} {proc} K{k}"; ./capstone-vexec ./{arm}/{p}.K{k}.dom {a.args} 2>/tmp/perf.err; rc=$?; '
                f'echo "@@PERF-END {n} {p} {arm} {proc} K{k} rc=$rc stderr_lines=$(wc -l </tmp/perf.err) '
                f'$(dmesg | grep \'capstone_vm: bench\' | tail -1 | sed \'s/.*capstone_vm: //\')"; rm -f /tmp/perf.err\n')
    f.write(f'echo "@@PERF-{a.name.upper()}-DONE"\n')
os.chmod(f'{a.out}/run-{a.name}.sh', 0o755)
print(len(steps), 'steps; order', progs)
