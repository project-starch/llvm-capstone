#!/usr/bin/env python3
"""One arm's manifest, and the QEMU functional table, from the builds and the run-qemu results.

  make-manifest.py --arm T1 --build <T1 build dir> --results <results.json> [--results ...]
                   --count-label T1c [--native-label native] [--excluded excluded.tsv]
                   --out manifest-T1.tsv [--sizing sizing-T1.tsv]

The results files are run-qemu.py outputs; later files override earlier ones per (label, image).
The arm's functional runs are read under the label --arm, its counting images under --count-label.

Manifest columns:
  image, program, suite, K, image_sha256, iterations (build-time size: BEEBS REPEAT_FACTOR, RV8 N,
  CoreMark ITERATIONS), reentrant, run_args (W I), own_verify, status, native_check_equal,
  timed_{malloc,free,realloc,calloc} (largest per timed iteration), run_{...} (the whole process at
  W=1 I=5), alloc_heavy, untimed_share.

  reentrant=yes when every W+I iteration of the program's K0 image in the --reentrancy-label arm
  (T1) verified with one check value and exited 0; otherwise no, and run_args is "0 1" (one iteration per process, the harness rule).
  leak_per_iteration: timed malloc + calloc - free, the bracket's view: rv8-qsort and rv8-miniz free
  their array after the end mark, so they read 1 here and 0 in run_leak.
  run_leak: run malloc + calloc - free over the whole process (what stays allocated at exit).
  alloc_heavy: yes >= 100 allocator calls per timed iteration, low 1-99, no 0. A T2/T1 near 1.0 on
  a "no" program says nothing about the cost of temporal safety.
  console_*_lines_per_process (console-audit.py over the functional run): lines the program printed
  besides the harness's BENCH-INFO/BENCH/BENCH-DONE, and lines capstone-vexec printed (its
  CAPSTONE_VM_ADD_OK per arena grant), per process at the functional run's W I.
  untimed_share: 1 - (guest ns inside all W+I brackets) / (guest ns of the whole process run), from
  an --icount run, where 1 guest ns is 1 retired instruction. Sizing data only, never a result.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

OWN_VERIFY_NONE = {'bs', 'compress', 'crc', 'cubic', 'fasta', 'fir', 'huffbench', 'lcdnum', 'miniz',
                   'minver', 'newlib-exp', 'newlib-log', 'newlib-mod', 'ns', 'picojpeg', 'qsort',
                   'qurt', 'rijndael', 'select', 'sglib-arrayheapsort', 'sglib-arrayquicksort',
                   'sglib-arraysort', 'sglib-dllist', 'sglib-hashtable', 'stb_perlin', 'ud',
                   'whetstone', 'wikisort'}


def suite_of(program):
    if program.startswith('ctl-'):
        return 'ctl'
    if program == 'coremark':
        return 'coremark'
    if program.startswith('rv8-'):
        return 'rv8'
    return 'beebs'


def own_verify(program):
    base = program[:-5] if program.endswith('-libc') else program
    if program.startswith('rv8-'):
        return 'driver'  # rv8-bench checks nothing; each driver checks what upstream main prints
    if program.startswith('ctl-'):
        return 'control'
    if program == 'coremark':
        return 'yes (CRCs for the list geometry)'
    return 'no' if base in OWN_VERIFY_NONE else 'yes'


def load(paths):
    merged, meta = {}, {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        for label, entries in d['results'].items():
            for image, e in entries.items():
                merged.setdefault(label, {})[image] = e
                meta[(label, image)] = d['meta']
    return merged, meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--results', type=Path, action='append', required=True)
    p.add_argument('--count-label', required=True)
    p.add_argument('--native-label', default='native')
    p.add_argument('--reentrancy-label', default='T1',
                   help="the arm whose K0 run decides reentrant (T1: a T2-only failure is a failure, not non-reentrancy)")
    p.add_argument('--excluded', type=Path)
    p.add_argument('--console', type=Path, action='append', default=[],
                   help='console-audit.py TSVs; later files override earlier ones per (set, image)')
    p.add_argument('--nonreentrant', type=Path, help='nonreentrant.txt (resize.py): these are reentrant=no')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sizing', type=Path)
    a = p.parse_args()
    res, meta = load(a.results)
    fr, cr, nr = res.get(a.arm, {}), res.get(a.count_label, {}), res.get(a.native_label, {})
    excluded = {}
    if a.excluded:
        for line in a.excluded.read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                n, why = line.split('\t', 1)
                excluded[n] = why
    rows = [l.split('\t') for l in (a.build / 'images.tsv').read_text().splitlines()[1:] if l.strip()]
    head = ['image', 'program', 'suite', 'K', 'image_sha256', 'iterations', 'reentrant', 'run_args',
            'own_verify', 'status', 'native_check_equal'] + \
           [f'timed_{k}' for k in ('malloc', 'free', 'realloc', 'calloc')] + \
           [f'run_{k}' for k in ('malloc', 'free', 'realloc', 'calloc')] + ['leak_per_iteration', 'run_leak', 'alloc_heavy', 'untimed_share', 'console_program_lines_per_process',
       'console_launcher_lines_per_process']
    out, sizing, warn = ['\t'.join(head)], ['\t'.join(['image', 'iterations', 'instr_per_timed_iteration_median',
                                                       'whole_run_instr', 'bracketed_instr', 'untimed_share',
                                                       'W', 'I'])], []
    console = {}
    for c in a.console:
        for line in c.read_text().splitlines()[1:]:
            f = line.split('\t')
            console[(f[0], f[1])] = (f[2], f[4])
    rr = res.get(a.reentrancy_label, {})
    nonre = set()
    if a.nonreentrant:
        nonre = {l.split()[0] for l in a.nonreentrant.read_text().splitlines() if l.strip() and not l.startswith('#')}
    for image, program, k, sha, iters in rows:
        e = fr.get(image)
        b = e['bench'] if e else []
        e0 = rr.get(f'{program}.K0')
        b0 = e0['bench'] if e0 else []
        ok_all = bool(b) and all(x['ok'] for x in b) and e['rc'] == 0
        checks = [x['check'] for x in b0]
        reentrant = 'yes' if b0 and all(x['ok'] for x in b0) and len(set(checks)) == 1 and e0['rc'] == 0 else 'no'
        if program in nonre:
            reentrant = 'no'
        if not e:
            status = 'NOT-RUN'
        elif not b:
            status = f'FAULT rc={e["rc"]}'
        elif ok_all:
            status = 'ok'
        elif e['rc'] not in (0, 1):
            status = f'FAULT after {len(b)} of {e["info"]["warmup"] + e["info"]["iterations"]} iterations rc={e["rc"]}'
        elif b[0]['ok']:
            status = f'ok-first-iteration-only rc={e["rc"]}'
        else:
            status = f'VERIFY-FAILED rc={e["rc"]}'
        if program in excluded:
            status = 'EXCLUDED: ' + excluded[program]
        n = nr.get(program)
        nchecks = [x['check'] for x in n['bench']] if n and n['bench'] else []
        if not nchecks or not b:
            neq = 'n/a'
        elif [x['check'] for x in b] == nchecks:
            neq = 'yes'
        else:
            neq = 'NO'
        c = cr.get(f'{program}.K0') or cr.get(program)
        timed = run = None
        if c and c['alloc']:
            per = [x for x in c['alloc'] if x['it'] != 'total' and not x['it'].startswith('w')]
            tot = [x for x in c['alloc'] if x['it'] == 'total']
            if per:
                timed = {kk: max(x[kk] for x in per) for kk in ('malloc', 'free', 'realloc', 'calloc')}
            if tot:
                run = tot[0]
        if timed is None:
            warn.append(f'{image}: no allocation counts under {a.count_label}')
        heavy = '?' if timed is None else ('yes' if sum(timed.values()) >= 100 else ('low' if sum(timed.values()) else 'no'))
        share = ''
        if e and b and e['wall'] and meta.get((a.arm, image), {}).get('icount'):
            bracketed = sum(x['ns'] for x in b)
            share = f'{1 - bracketed / e["wall"]:.3f}'
            med = statistics.median(x['ns'] for x in b if not x['it'].startswith('w'))
            sizing.append('\t'.join(map(str, [image, iters, int(med), e['wall'], bracketed, share,
                                              e['info']['warmup'], e['info']['iterations']])))
        f = lambda d, kk: str(d[kk]) if d else ''
        out.append('\t'.join([image, program, suite_of(program), k, sha, iters, reentrant,
                              '1 5' if reentrant == 'yes' else '0 1', own_verify(program), status, neq] +
                             [f(timed, kk) for kk in ('malloc', 'free', 'realloc', 'calloc')] +
                             [f(run, kk) for kk in ('malloc', 'free', 'realloc', 'calloc')] +
                             [str(timed['malloc'] + timed['calloc'] - timed['free']) if timed else '',
                              str(run['malloc'] + run['calloc'] - run['free']) if run else '', heavy, share] +
                             list(console.get((a.arm, image), ('', '')))))
    a.out.write_text('\n'.join(out) + '\n')
    if a.sizing:
        a.sizing.write_text('\n'.join(sizing) + '\n')
    for w in warn:
        print('WARNING', w, file=sys.stderr)
    print(f'{a.out}: {len(rows)} images')
    return 0


if __name__ == '__main__':
    sys.exit(main())
