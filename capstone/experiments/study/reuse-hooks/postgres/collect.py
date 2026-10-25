#!/usr/bin/env python3
"""Collect the PostgreSQL reuse-distance row from the two boots' serial logs.

  collect.py <run dir> <kit dir> <reuse.json out>

<run dir> holds runs/pg-<arm>/serial.log, stages/pg-<arm>/{<arm>/bin/postgres,gate.sh,work.sql}
and out/pg-<arm>/source/domain/postgresql-17.5/src/backend/utils/mmgr/mcxt.c. Exits nonzero,
writing nothing, if a run did not complete, exited nonzero, logged a domain fault, lacks the
REUSE-GAP line, reports a nonzero observer error, or does not give the 22 work.sql rows of
the same-answers run (run-ports.py's pg_rows, sha256 280000084d6b...).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROWS_SHA256 = '280000084d6bc8493f3c91444aed0fc7d3d09d0c38e05030deb4ec809d25789c'
GAP = re.compile(r'^REUSE-GAP contexts attempts=(\d+) issues=(\d+) releases=(\d+) reuses=(\d+) '
                 r'distinct=(\d+) capacity=(\d+) error=(\d+) reset_releases=(\d+) resizes=(\d+) '
                 r'bins=([\d,]+)$', re.M)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(run, arm, suffix=''):
    log = (run / f'runs/pg-{arm}{suffix}/serial.log').read_text(errors='replace').replace('\r', '')
    tag = f'postgres-{arm}'
    m = re.search(rf'^PORT_BEGIN:{tag}\n(.*?)^PORT_WALL:{tag}:(\d+)\n^PORT_END:{tag}:(\d+)$',
                  log, re.M | re.S)
    if not m:
        sys.exit(f'{arm}: no PORT_BEGIN/END block')
    body = m.group(1)
    rows = re.findall(r'\t(?: \d+: [^\n]+|----)', body)  # run-ports.py pg_rows
    r = dict(exit=int(m.group(3)), guest_wall_s=int(m.group(2)),
             domain_fault='domain fault' in body,
             completed='\nVIRTUAL_STAGED_DONE\n' in log,
             rmmod=int(re.search(r'^RMMOD:(\d+)$', log, re.M).group(1)),
             rows=len(rows),
             rows_sha256=hashlib.sha256(('\n'.join(rows) + '\n').encode()).hexdigest())
    after = re.search(r'^STATS_AFTER:(\{.*\})$', log, re.M)
    r['launcher_stats_after'] = json.loads(after.group(1)) if after else None
    vm = re.search(r'^CAPSTONE_VM_STATS (.*)$', body, re.M)
    r['process_vm_stats'] = vm.group(1) if vm else None
    gaps = GAP.findall(body)
    if len(gaps) != 1:
        sys.exit(f'{arm}: {len(gaps)} REUSE-GAP contexts lines')
    g = gaps[0]
    a, i, rel, reu, dist, cap, err, rr, rs = map(int, g[:9])
    bins = list(map(int, g[9].split(',')))
    if len(bins) != 32 or sum(bins) != reu:
        sys.exit(f'{arm}: bins do not reconcile with reuses')
    r['contexts'] = dict(buckets={str(k + 1): v for k, v in enumerate(bins) if v},
                         issues=i, releases=rel, reuses=reu, distinct_starts=dist,
                         attempts=a, error=err)
    r['extra'] = dict(capacity=cap, reset_releases=rr, pfree_and_moving_realloc_releases=rel - rr,
                      in_place_resizes=rs)
    return r


def check(name, r):
    bad = [k for k, ok in (('exit', r['exit'] == 0), ('fault', not r['domain_fault']),
                           ('completed', r['completed']), ('rmmod', r['rmmod'] == 0),
                           ('rows', r['rows'] == 22 and r['rows_sha256'] == ROWS_SHA256),
                           ('observer', r['contexts']['error'] == 0)) if not ok]
    if bad:
        sys.exit(f'{name}: rejected: {bad}')


def main():
    run, kit, out = map(Path, sys.argv[1:4])
    arms = {arm: parse(run, arm) for arm in ('stock', 'protected')}
    for arm, r in arms.items():
        check(arm, r)
    # Repetitions 2 and 3 (run-reps.sh): a variation check beside the planned boot.
    reps = {}
    for arm in arms:
        reps[arm] = []
        for n in (2, 3):
            r = parse(run, arm, f'-r{n}')
            check(f'{arm} r{n}', r)
            reps[arm].append(dict(rep=n, guest_wall_s=r['guest_wall_s'],
                                  boundaries=dict(contexts=r['contexts']), contexts_detail=r['extra']))
    mcxt = 'out/pg-{}/source/domain/postgresql-17.5/src/backend/utils/mmgr/mcxt.c'
    res = dict(
        program='PostgreSQL 17.5',
        pre_registration='16d2171a3edc',
        workload='work.sql in the single-user backend (run-ports.py command)',
        boundaries=dict(contexts={arm: r['contexts'] for arm, r in arms.items()}),
        contexts_detail={arm: r['extra'] for arm, r in arms.items()},
        repetitions=reps,
        repetitions_identical_to_planned_boot={
            arm: all(x['boundaries']['contexts'] == arms[arm]['contexts'] for x in reps[arm])
            for arm in arms},
        images={arm: sha(run / f'stages/pg-{arm}/{arm}/bin/postgres') for arm in arms},
        mcxt_sources={arm: sha(run / mcxt.format(arm)) for arm in arms},
        sanity={arm: dict(exit=r['exit'], domain_fault=r['domain_fault'], completed=r['completed'],
                          rmmod=r['rmmod'], rows=r['rows'], rows_sha256=r['rows_sha256'],
                          rows_equal_same_answers=r['rows_sha256'] == ROWS_SHA256,
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
            'work.sql': sha(run / 'stages/pg-stock/work.sql'),
            'gate.sh stock': sha(run / 'stages/pg-stock/gate.sh'),
            'gate.sh protected': sha(run / 'stages/pg-protected/gate.sh')})
    out.write_text(json.dumps(res, indent=1) + '\n')
    for arm in arms:
        g = res['boundaries']['contexts'][arm]
        print('contexts', arm, {k: g[k] for k in ('attempts', 'issues', 'releases', 'reuses',
                                                 'distinct_starts', 'error')},
              res['contexts_detail'][arm], g['buckets'])


if __name__ == '__main__':
    main()
