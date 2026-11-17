#!/usr/bin/env python3
"""mk-p1kits.py: the P1 embedded-suite kits (BEEBS, RV8, CoreMark; 103 programs) for bkit91, from the bitstream-3 kits.

Sources: perf/main3b/t3k1..3 (the 15 allocating programs) and perf/main5/t3a1..4 (the other 88). Each P1 kit keeps its
source kit's process order with every T2 step removed (T1 and T3 only; W/I as in the source), renumbered from 0; the
source step number is kept in steps.tsv's last column. T1 = FRCDKMSspa3 SDK, T3 = FRCDKMSa3 SDK
(perf/p1suite/build/{T1,T3}, harness -DBENCH_CONTEXT_COUNTERS).

Launch: the native route of bkit91 has no capstone-vexec and no /dev/capstone-vm, so each process is a direct exec of
its .dom through the kernel's binfmt, as the SPEC native stages run: images copied to /tmp/nt/b (tmpfs), cwd
/tmp/nt/sr (dom_grant_root), stderr to tmpfs. Around each process the run script prints the module's native_counters
(@@NATIVE-CNT <n> begin/end, the SPEC stages' format): the per-process CSRUNV-window cross-check (run_cycles, runs).

Layout: a native stage <work>/<kit>-ns (stage/run.sh, stage/bin/native/{T1,T3}/*.dom, plan.json), then the kit's own
make-train-stage.sh makes the board stage <out>/<kit> (the native smoke, train/ = the suite, spec.sh). Each board stage
gets <out>/<kit>.SHA256SUMS over every file. Refuses to overwrite."""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

H = Path(os.environ.get('CAPSTONE_WORK', f"/mnt/hdd/{os.environ['USER']}/capstone")); P = H / 'perf/p1suite'
KIT = H / 't3-smoke/bkit91-k2n-native-p1-tree-cscfix'
SRC = [(f'p1k{i}', H / f'perf/main3b/t3k{i}') for i in (1, 2, 3)] + [(f'p1a{i}', H / f'perf/main5/t3a{i}') for i in (1, 2, 3, 4)]
BUILD = {'T1': P / 'build/T1', 'T3': P / 'build/T3'}
WORK, OUT = P / 'ns', P / 'kits'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def steps_of(src):
    rows = [l.rstrip('\n').split('\t') for l in open(src / 'steps.tsv')]
    assert rows[0] == ['n', 'program', 'arm', 'proc', 'K', 'args'], rows[0]
    return rows[1:]


def run_script(kit, rows):
    progs = sorted({r[1] for r in rows})
    L = ['#!/bin/sh',
         f'# P1 embedded suite, kit {kit}: {len(progs)} programs, {len(rows)} processes (T1, T3), from the bitstream-3 kit\'s',
         '# order without T2. Each process is a direct exec of its .dom (binfmt), cwd /tmp/nt/sr, stderr to tmpfs; console',
         '# gets stdout, the line count, and native_counters before and after (@@NATIVE-CNT).',
         'S=$(cd "$(dirname "$0")" && pwd)',
         'cnt() { cat /sys/module/capstone_vm/parameters/native_counters 2>/dev/null || echo absent; }',
         'mkdir -p /tmp/nt && { mountpoint -q /tmp/nt || mount -t tmpfs tmpfs /tmp/nt; } || exit 1',
         'cd / && rm -rf /tmp/nt/sr /tmp/nt/b && mkdir -p /tmp/nt/sr /tmp/nt/b/T1 /tmp/nt/b/T3 && '
         'cp "$S"/bin/native/T1/* /tmp/nt/b/T1/ && cp "$S"/bin/native/T3/* /tmp/nt/b/T3/ && chmod 755 /tmp/nt/b/T1/* /tmp/nt/b/T3/* && '
         'cd /tmp/nt/sr && mkdir -m 0700 .capstone-tmp || { echo "@@PERF-SETUP-FAILED"; exit 1; }',
         f'echo "@@PERF-{kit.upper()}-BEGIN processes={len(rows)} $(cnt)"']
    for n, (src_n, prog, arm, proc, k, args) in enumerate(rows):
        tag = f'{n} {prog} {arm} {proc} K{k}'
        L.append(f'echo "@@PERF-BEGIN {tag}"; echo "@@NATIVE-CNT {n} begin $(cnt)"; '
                 f'/tmp/nt/b/{arm}/{prog}.K{k}.dom {args} 2>/tmp/nt/perf.err; rc=$?; '
                 f'echo "@@NATIVE-CNT {n} end $(cnt)"; '
                 f'echo "@@PERF-END {tag} rc=$rc stderr_lines=$(wc -l </tmp/nt/perf.err)"; rm -f /tmp/nt/perf.err')
    L += ['cd / && rm -rf /tmp/nt/sr /tmp/nt/b', f'echo "@@PERF-{kit.upper()}-DONE"']
    return '\n'.join(L) + '\n'


def main():
    only = set(sys.argv[1:])
    WORK.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    for kit, src in SRC:
        if only and kit not in only:
            continue
        ns, brd = WORK / f'{kit}-ns', OUT / kit
        if ns.exists() or brd.exists():
            sys.exit(f'STOP: {ns} or {brd} exists')
        rows = [r for r in steps_of(src) if r[2] != 'T2']
        assert all(r[2] in ('T1', 'T3') for r in rows)
        # the T1/T3 order of the source is kept: each arm's sequence equals the source's with T2 dropped
        st = ns / 'stage'; (st / 'bin/native/T1').mkdir(parents=True); (st / 'bin/native/T3').mkdir(parents=True)
        used = sorted({(r[2], r[1], r[4]) for r in rows})
        for arm, prog, k in used:
            img = BUILD[arm] / f'{prog}.K{k}.dom'
            assert img.is_file(), img
            dst = st / 'bin/native' / arm / img.name
            shutil.copyfile(img, dst); os.chmod(dst, 0o755)
        (st / 'run.sh').write_text(run_script(kit, rows))
        with open(ns / 'steps.tsv', 'w') as f:
            f.write('n\tprogram\tarm\tproc\tK\targs\tsrc_n\n')
            for n, r in enumerate(rows):
                f.write('\t'.join([str(n), r[1], r[2], r[3], r[4], r[5], r[0]]) + '\n')
        plan = {'kit': kit, 'source_kit': str(src), 'processes': len(rows), 'programs': len({r[1] for r in rows}),
                'arms': {'T1': {'sdk': str(H / 'perf/cclever/arms/FRCDKMSspa3/sdk'), 'build': str(BUILD['T1'])},
                         'T3': {'sdk': str(H / 'perf/cclever/arms/FRCDKMSa3/sdk'), 'build': str(BUILD['T3'])}},
                'images': {f'{arm}/{prog}.K{k}.dom': sha(st / 'bin/native' / arm / f'{prog}.K{k}.dom') for arm, prog, k in used}}
        (ns / 'plan.json').write_text(json.dumps(plan, indent=1) + '\n')
        o = subprocess.run(['bash', str(KIT / 'make-train-stage.sh'), str(ns), str(brd)], capture_output=True, text=True)
        assert o.returncode == 0, o.stdout + o.stderr
        shutil.copyfile(ns / 'steps.tsv', OUT / f'{kit}.steps.tsv')
        files = sorted(str(p.relative_to(brd)) for p in brd.rglob('*') if p.is_file())
        with open(OUT / f'{kit}.SHA256SUMS', 'w') as f:
            for name in files:
                f.write(f'{sha(brd / name)}  ./{name}\n')
        print(kit, len(rows), 'processes', len(used), 'images', o.stdout.strip())


main()
