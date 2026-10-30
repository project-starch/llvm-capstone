#!/usr/bin/env python3
"""Split the main run's programs into board kits and write each kit's stage: images, the run script, steps.tsv.

  make-kits.py --images <build dir with T1/ T2/> --programs <file, one name per line> --out <dir> --kits N
               [--procs 4 --args "1 5"] [--single crc32,huffbench,... --single-procs 11 --single-args "0 1"]
               [--cap-mb 24] [--seed 20261026] [--prefix b] [--arms T1,T3,T2 --arm-dirs T3=<build dir>]
               [--nonreentrant <list>] [--plain Plain=<build dir> --probe <plain-tick-probe> --strip <llvm-objcopy>]

--arms names the arms in order (default T1,T2; the T3 kits used T1,T3,T2); every arm of a program runs in the same
kit, alternating ABC..CBA. --arm-dirs takes an arm's images from another build root.

--plain NAME=DIR adds an arm of plain (non-Capstone) executables, DIR/<program> from `build.py --mode plain`: it
is listed in --arms like any other arm, runs directly (no launcher), and its END line carries no dmesg
quote (the module never sees it). Its images are DIR/<program>.K<k> (DIR/images.tsv): every Plain program is
linked at layout pads K0..K3, and its processes take them in turn, as an allocating program's Capstone processes do. Each copy is stripped with --strip (llvm-objcopy --strip-all; the kit's
plain-images.tsv records the build's and the stripped copy's sha256). --probe stages that program and runs it once,
before the first step, between @@PLAINPROBE-BEGIN/END markers.

--every-kit a,b puts those programs (the controls) into EVERY kit, after the greedy split of the others, so each
boot carries its own calibration (the Plain arm's per-tick cost from ctl-loop, each arm's bracket cost from
ctl-empty). Their rows then come from several boots; analyse them per boot.

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
ap.add_argument('--every-kit', default='');
ap.add_argument('--plain', action='append', default=[]); ap.add_argument('--probe'); ap.add_argument('--strip')
ap.add_argument('--nonreentrant', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nonreentrant.txt'))
a = ap.parse_args()
ARMS = a.arms.split(',')
DIRS = dict(x.split('=') for x in a.arm_dirs.split(',') if x) # arm -> images root
PLAIN = dict(x.split('=', 1) for x in a.plain)                   # arm -> plain build dir
if set(PLAIN) - set(ARMS): raise SystemExit(f'--plain arm(s) {sorted(set(PLAIN) - set(ARMS))} not in --arms')
if PLAIN and not a.strip: raise SystemExit('--plain needs --strip (llvm-objcopy)')
import hashlib, subprocess
def sha(path): return hashlib.sha256(open(path, 'rb').read()).hexdigest()
def src_of(arm, p, k): return f'{PLAIN[arm]}/{p}.K{k}' if arm in PLAIN else f'{DIRS.get(arm, a.images)}/{arm}/{p}.K{k}.dom'
def dst_of(stage, arm, p, k): return f'{stage}/{arm}/{p}.K{k}' if arm in PLAIN else f'{stage}/{arm}/{p}.K{k}.dom'
pks = {}   # (plain arm, program) -> its K list
for arm, d in PLAIN.items():
    for r in csv.DictReader(open(f'{d}/images.tsv'), delimiter='\t'):
        pks.setdefault((arm, r['program']), []).append(int(r['K']))
def klist(arm, p): return sorted(pks[(arm, p)]) if arm in PLAIN else (ks[p] if len(ks[p]) > 1 else [0])

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
    size = sum(os.path.getsize(src_of(arm, p, k)) // (5 if arm in PLAIN else 1) for arm in ARMS   # stripped ~1/5
               for k in klist(arm, p))
    return procs, args, t, size

info = {p: plan(p) for p in progs}
kits = [dict(progs=[], t=0.0, size=0) for _ in range(a.kits)]
every = [p for p in filter(None, a.every_kit.split(',')) if p in info]
if len(every) != len(list(filter(None, a.every_kit.split(',')))): raise SystemExit(f'--every-kit {a.every_kit}: not all in --programs')
for p in sorted((p for p in progs if p not in every), key=lambda p: -info[p][2]):
    cands = [k for k in kits if k['size'] + info[p][3] <= a.cap_mb * 1e6]
    if not cands: raise SystemExit(f'{p}: no kit has room under {a.cap_mb} MB')
    k = min(cands, key=lambda k: k['t'])
    k['progs'].append(p); k['t'] += info[p][2]; k['size'] += info[p][3]

for k in kits:
    for p in every:
        k['progs'].append(p); k['t'] += info[p][2]; k['size'] += info[p][3]
for ki, k in enumerate(kits):
    name = f'{a.prefix}{ki + 1}'
    stage = f'{a.out}/{name}/stage'
    if os.path.exists(f'{a.out}/{name}'): shutil.rmtree(f'{a.out}/{name}')
    [os.makedirs(f'{stage}/{x}') for x in ARMS]
    shutil.copy(a.launcher, f'{stage}/capstone-vexec')
    if a.probe: shutil.copy(a.probe, f'{stage}/plain-tick-probe')
    plain_rows = []
    order = list(k['progs']); random.Random(a.seed + ki).shuffle(order)
    steps = []
    for pos, p in enumerate(order):
        procs, args, _, _ = info[p]
        unit = ARMS + ARMS[::-1] if pos % 2 == 0 else ARMS[::-1] + ARMS
        pat = (unit * procs)[:len(ARMS) * procs]
        cnt = {x: 0 for x in ARMS}
        for arm in pat:
            kl = klist(arm, p)
            kk = kl[cnt[arm] % len(kl)]
            steps.append((p, arm, cnt[arm], kk, args)); cnt[arm] += 1
            dst = dst_of(stage, arm, p, kk)
            if not os.path.exists(dst):
                if arm in PLAIN:
                    subprocess.run([a.strip, '--strip-all', src_of(arm, p, kk), dst], check=True)
                    plain_rows.append((arm, p, sha(src_of(arm, p, kk)), sha(dst)))
                else:
                    shutil.copy(src_of(arm, p, kk), dst)
    with open(f'{a.out}/{name}/steps.tsv', 'w') as f:
        f.write('n\tprogram\tarm\tproc\tK\targs\n')
        for n, s in enumerate(steps): f.write(f'{n}\t{s[0]}\t{s[1]}\t{s[2]}\t{s[3]}\t{s[4]}\n')
    if plain_rows:
        with open(f'{a.out}/{name}/plain-images.tsv', 'w') as f:
            f.write('arm\tprogram\tbuild_sha256\tstaged_sha256\n')
            for r in plain_rows: f.write('\t'.join(r) + '\n')
    with open(f'{stage}/run-{name}.sh', 'w') as f:
        f.write(f'#!/bin/sh\n# perf main run, kit {name}: {len(order)} programs, {len(steps)} processes, seed {a.seed + ki}. '
                'stderr of every process to tmpfs; console gets stdout and the line count only.\n')
        f.write('cd "$(dirname "$0")" || exit 1\nls /sys/module/capstone_vm/parameters | grep bench\n')
        if a.probe:
            f.write('echo "@@PLAINPROBE-BEGIN"; ./plain-tick-probe; echo "@@PLAINPROBE-END rc=$?"\n')
        for n, (p, arm, proc, kk, args) in enumerate(steps):
            if arm in PLAIN:
                f.write(f'echo "@@PERF-BEGIN {n} {p} {arm} {proc} K{kk}"; ./{arm}/{p}.K{kk} {args} 2>/tmp/perf.err; rc=$?; '
                        f'echo "@@PERF-END {n} {p} {arm} {proc} K{kk} rc=$rc stderr_lines=$(wc -l </tmp/perf.err)"; '
                        'rm -f /tmp/perf.err\n')
                continue
            f.write(f'echo "@@PERF-BEGIN {n} {p} {arm} {proc} K{kk}"; ./capstone-vexec ./{arm}/{p}.K{kk}.dom {args} 2>/tmp/perf.err; rc=$?; '
                    f'echo "@@PERF-END {n} {p} {arm} {proc} K{kk} rc=$rc stderr_lines=$(wc -l </tmp/perf.err) '
                    f'$(dmesg | grep \'capstone_vm: bench\' | tail -1 | sed \'s/.*capstone_vm: //\')"; rm -f /tmp/perf.err\n')
        f.write(f'echo "@@PERF-{name.upper()}-DONE"\n')
    os.chmod(f'{stage}/run-{name}.sh', 0o755)
    print(f'{name}: {len(order)} programs, {len(steps)} processes, est {k["t"] / 3600:.2f} h, {k["size"] / 1e6:.1f} MB')
