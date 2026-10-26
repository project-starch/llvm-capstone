#!/usr/bin/env python3
"""Split the main run's programs into board kits and write each kit's stage: images, the run script, steps.tsv.

  make-kits.py --images <build dir with T1/ T2/> --programs <file, one name per line> --out <dir> --kits N
               [--procs 4 --args "1 5"] [--single crc32,huffbench,... --single-procs 11 --single-args "0 1"]
               [--cap-mb 24] [--seed 20261026] [--prefix b] [--arms T1,T3,T2 --arm-dirs T3=<build dir>]
               [--nonreentrant <list>]

--arms names the arms in order (default T1,T2; the T3 kits used T1,T3,T2); every arm of a program runs in the same
kit, alternating ABC..CBA. --arm-dirs takes an arm's images from another build root.

A program in --nonreentrant (default: ../nonreentrant.txt beside this directory, when present) must be in --single,
or the script refuses: on 2026-10-27 a T3 kit was built without --single and ran huffbench-libc at five iterations
per process; it was caught before any boot.

Kits are filled greedily by expected board time (longest first, onto the kit with the least time), with every kit's
stage under --cap-mb. A program's both arms and all its processes go into ONE kit (same board, same boot). Within a
kit, programs run in a fixed random order (seed + kit index); each program's processes alternate arms ABBA, starting
with T1 for even positions and T2 for odd. An allocating program (more than one K image) spreads its processes over
its K variants in turn. Each process's stderr goes to tmpfs (the console is a 115,200-baud UART).
"""
import argparse, csv, os, random, shutil

ap = argparse.ArgumentParser()
ap.add_argument('--images', required=True); ap.add_argument('--programs', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--kits', type=int, required=True)
ap.add_argument('--procs', type=int, default=4); ap.add_argument('--args', default='1 5')
ap.add_argument('--single', default=''); ap.add_argument('--single-procs', type=int, default=11)
ap.add_argument('--single-args', default='0 1')
ap.add_argument('--cap-mb', type=float, default=24.0); ap.add_argument('--seed', type=int, default=20261026)
ap.add_argument('--c-proc', type=float, default=10.7); ap.add_argument('--c-iter', type=float, default=1.6)
ap.add_argument('--prefix', default='k'); ap.add_argument('--arms', default='T1,T2'); ap.add_argument('--arm-dirs', default=''); ap.add_argument('--launcher', required=True)
ap.add_argument('--nonreentrant', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nonreentrant.txt'))
a = ap.parse_args()
ARMS = a.arms.split(',')
DIRS = dict(x.split('=') for x in a.arm_dirs.split(',') if x) # arm -> images root

progs = [l.split()[0] for l in open(a.programs) if l.strip() and not l.startswith('#')]
single = set(filter(None, a.single.split(',')))
if os.path.exists(a.nonreentrant):
    nonre = {l.split()[0] for l in open(a.nonreentrant) if l.strip() and not l.startswith('#')}
    missing = sorted((nonre & set(progs)) - single)
    if missing: raise SystemExit(f'not re-entrant but not in --single: {missing} ({a.nonreentrant})')
rows = list(csv.DictReader(open(f'{a.images}/T1/images.tsv'), delimiter='\t'))
ks = {}
for r in rows:
    ks.setdefault(r['program'], []).append(int(r['K']))
for p in progs:
    if p not in ks: raise SystemExit(f'{p}: no image in {a.images}')
    ks[p].sort()

def plan(p):
    procs, args = (a.single_procs, a.single_args) if p in single else (a.procs, a.args)
    w, i = map(int, args.split())
    t = len(ARMS) * procs * (a.c_proc + (w + i) * a.c_iter)
    size = sum(os.path.getsize(f'{DIRS.get(arm, a.images)}/{arm}/{p}.K{k}.dom') for arm in ARMS
               for k in (ks[p] if len(ks[p]) > 1 else [0]))
    return procs, args, t, size

info = {p: plan(p) for p in progs}
kits = [dict(progs=[], t=0.0, size=0) for _ in range(a.kits)]
for p in sorted(progs, key=lambda p: -info[p][2]):
    cands = [k for k in kits if k['size'] + info[p][3] <= a.cap_mb * 1e6]
    if not cands: raise SystemExit(f'{p}: no kit has room under {a.cap_mb} MB')
    k = min(cands, key=lambda k: k['t'])
    k['progs'].append(p); k['t'] += info[p][2]; k['size'] += info[p][3]

for ki, k in enumerate(kits):
    name = f'{a.prefix}{ki + 1}'
    stage = f'{a.out}/{name}/stage'
    if os.path.exists(f'{a.out}/{name}'): shutil.rmtree(f'{a.out}/{name}')
    [os.makedirs(f'{stage}/{x}') for x in ARMS]
    shutil.copy(a.launcher, f'{stage}/capstone-vexec')
    order = list(k['progs']); random.Random(a.seed + ki).shuffle(order)
    steps = []
    for pos, p in enumerate(order):
        procs, args, _, _ = info[p]
        unit = ARMS + ARMS[::-1] if pos % 2 == 0 else ARMS[::-1] + ARMS
        pat = (unit * procs)[:len(ARMS) * procs]
        cnt = {x: 0 for x in ARMS}
        for arm in pat:
            kk = ks[p][cnt[arm] % len(ks[p])] if len(ks[p]) > 1 else 0
            steps.append((p, arm, cnt[arm], kk, args)); cnt[arm] += 1
            dst = f'{stage}/{arm}/{p}.K{kk}.dom'
            if not os.path.exists(dst): shutil.copy(f'{DIRS.get(arm, a.images)}/{arm}/{p}.K{kk}.dom', dst)
    with open(f'{a.out}/{name}/steps.tsv', 'w') as f:
        f.write('n\tprogram\tarm\tproc\tK\targs\n')
        for n, s in enumerate(steps): f.write(f'{n}\t{s[0]}\t{s[1]}\t{s[2]}\t{s[3]}\t{s[4]}\n')
    with open(f'{stage}/run-{name}.sh', 'w') as f:
        f.write(f'#!/bin/sh\n# perf main run, kit {name}: {len(order)} programs, {len(steps)} processes, seed {a.seed + ki}. '
                'stderr of every process to tmpfs; console gets stdout and the line count only.\n')
        f.write('cd "$(dirname "$0")" || exit 1\nls /sys/module/capstone_vm/parameters | grep bench\n')
        for n, (p, arm, proc, kk, args) in enumerate(steps):
            f.write(f'echo "@@PERF-BEGIN {n} {p} {arm} {proc} K{kk}"; ./capstone-vexec ./{arm}/{p}.K{kk}.dom {args} 2>/tmp/perf.err; rc=$?; '
                    f'echo "@@PERF-END {n} {p} {arm} {proc} K{kk} rc=$rc stderr_lines=$(wc -l </tmp/perf.err) '
                    f'$(dmesg | grep \'capstone_vm: bench\' | tail -1 | sed \'s/.*capstone_vm: //\')"; rm -f /tmp/perf.err\n')
        f.write(f'echo "@@PERF-{name.upper()}-DONE"\n')
    os.chmod(f'{stage}/run-{name}.sh', 0o755)
    print(f'{name}: {len(order)} programs, {len(steps)} processes, est {k["t"] / 3600:.2f} h, {k["size"] / 1e6:.1f} MB')
