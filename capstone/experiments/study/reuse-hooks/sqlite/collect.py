#!/usr/bin/env python3
"""Collect the SQLite reuse-distance row from the two boots' serial logs.

  collect.py <run dir> <kit dir> <phases.json of the same-answers run> <reuse.json out>

<run dir> holds runs/sqlite-<arm>/serial.log, stages/sqlite-<arm>/{speedtest1.dom,gate.sh}
and sq/<arm>/src/sqlite3-capstone.c. Exits nonzero, writing nothing, if a run did not
complete, exited nonzero, logged a domain fault, lacks a REUSE-GAP line, reports a nonzero
observer error, or gives phases that differ from the same-answers native phases.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ORA = re.compile(r'^STUDY-ORACLE phase=(\d+) rows=(\d+) hash=([0-9a-f]{16})$', re.M)
GAP = re.compile(r'^REUSE-GAP (\w+) attempts=(\d+) issues=(\d+) releases=(\d+) reuses=(\d+) '
                 r'distinct=(\d+) capacity=(\d+) error=(\d+) bins=([\d,]+)$', re.M)
BOUNDARIES = ('memsys5', 'lookaside')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(run, arm, suffix=''):
    log = (run / f'runs/sqlite-{arm}{suffix}/serial.log').read_text(errors='replace').replace('\r', '')
    tag = f'sqlite-{arm}'
    m = re.search(rf'^PORT_BEGIN:{tag}\n(.*?)^PORT_WALL:{tag}:(\d+)\n^PORT_END:{tag}:(\d+)$',
                  log, re.M | re.S)
    if not m:
        sys.exit(f'{arm}: no PORT_BEGIN/END block')
    body = m.group(1)
    r = dict(exit=int(m.group(3)), guest_wall_s=int(m.group(2)),
             domain_fault='domain fault' in body,
             completed='\nVIRTUAL_STAGED_DONE\n' in log,
             rmmod=int(re.search(r'^RMMOD:(\d+)$', log, re.M).group(1)),
             phases=[dict(phase=int(p), rows=int(n), hash=h) for p, n, h in ORA.findall(body)])
    after = re.search(r'^STATS_AFTER:(\{.*\})$', log, re.M)
    r['launcher_stats_after'] = json.loads(after.group(1)) if after else None
    vm = re.search(r'^CAPSTONE_VM_STATS (.*)$', body, re.M)
    r['process_vm_stats'] = vm.group(1) if vm else None
    gaps = {}
    for g in GAP.findall(body):
        name = g[0]
        if name in gaps:
            sys.exit(f'{arm}: two REUSE-GAP {name} lines')
        a, i, rel, reu, dist, cap, err = map(int, g[1:8])
        bins = list(map(int, g[8].split(',')))
        if len(bins) != 32 or sum(bins) != reu:
            sys.exit(f'{arm} {name}: bins do not reconcile with reuses')
        gaps[name] = dict(buckets={str(k + 1): v for k, v in enumerate(bins) if v},
                          issues=i, releases=rel, reuses=reu, distinct_starts=dist,
                          attempts=a, error=err, capacity=cap)
    r['boundaries'] = gaps
    return r


def check(name, r, native):
    bad = [k for k, ok in (('exit', r['exit'] == 0), ('fault', not r['domain_fault']),
                           ('completed', r['completed']), ('rmmod', r['rmmod'] == 0),
                           ('phases', r['phases'] == native)) if not ok]
    for b in BOUNDARIES:
        g = r['boundaries'].get(b)
        if g is None or g['error'] != 0:
            bad.append(f'{b} observer')
    if bad:
        sys.exit(f'{name}: rejected: {bad}')


def main():
    run, kit, phases_json, out = map(Path, sys.argv[1:5])
    native = json.loads(phases_json.read_text())['native']['phases']
    arms = {arm: parse(run, arm) for arm in ('stock', 'protected')}
    for arm, r in arms.items():
        check(arm, r, native)
    # Repetitions 2 and 3 (run-reps.sh): a variation check beside the planned boot.
    reps = {}
    for arm in arms:
        reps[arm] = []
        for n in (2, 3):
            r = parse(run, arm, f'-r{n}')
            check(f'{arm} r{n}', r, native)
            reps[arm].append(dict(rep=n, guest_wall_s=r['guest_wall_s'], boundaries={
                b: {k: v for k, v in r['boundaries'][b].items() if k != 'capacity'}
                for b in BOUNDARIES}))
    res = dict(
        program='SQLite 3.22.0',
        pre_registration='16d2171a3edc',
        workload='speedtest1 --heap 8388608 64 --size 1 st.db (main, 8 MiB memsys5 heap)',
        boundaries={b: {arm: {k: v for k, v in arms[arm]['boundaries'][b].items() if k != 'capacity'}
                        for arm in arms} for b in BOUNDARIES},
        observer_capacity={b: arms['stock']['boundaries'][b]['capacity'] for b in BOUNDARIES},
        repetitions=reps,
        repetitions_identical_to_planned_boot={
            arm: all(x['boundaries'] == {b: {k: v for k, v in arms[arm]['boundaries'][b].items()
                                             if k != 'capacity'} for b in BOUNDARIES}
                     for x in reps[arm]) for arm in arms},
        images={arm: sha(run / f'stages/sqlite-{arm}/speedtest1.dom') for arm in arms},
        engine_sources={arm: sha(run / f'sq/{arm}/src/sqlite3-capstone.c') for arm in arms},
        sanity={arm: dict(exit=r['exit'], domain_fault=r['domain_fault'], completed=r['completed'],
                          rmmod=r['rmmod'], phases=len(r['phases']),
                          phases_equal_same_answers_native=r['phases'] == native,
                          guest_wall_s=r['guest_wall_s'],
                          process_vm_stats=r['process_vm_stats'],
                          launcher_stats_after=r['launcher_stats_after']) for arm, r in arms.items()},
        platform={
            'qemu-system-riscv64': sha(kit / 'kit-sublet/qemu-system-riscv64'),
            'capstone-vexec': sha(kit / 'kit-sublet/adapter/capstone-vexec'),
            'capstone_vm.ko': sha(kit / 'kit-sublet/adapter/module/capstone_vm.ko'),
            'Image': sha(kit / 'kit-sublet/images/Image'),
            'fw_jump.elf': sha(kit / 'kit-sublet/images/fw_jump.elf'),
            'rootfs.ext2': sha(kit / 'kit-sublet/images/rootfs.ext2'),
            'clang': sha(kit / 'toolchain/bin/clang'),
            'gate.sh stock': sha(run / 'stages/sqlite-stock/gate.sh'),
            'gate.sh protected': sha(run / 'stages/sqlite-protected/gate.sh')})
    out.write_text(json.dumps(res, indent=1) + '\n')
    for b in BOUNDARIES:
        for arm in arms:
            g = res['boundaries'][b][arm]
            print(b, arm, {k: g[k] for k in ('attempts', 'issues', 'releases', 'reuses',
                                            'distinct_starts', 'error')}, g['buckets'])


if __name__ == '__main__':
    main()
