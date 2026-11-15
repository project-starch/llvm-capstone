#!/usr/bin/env python3
"""One boot of the PostgreSQL reuse-distance measurement on pgbench's TPC-B-like workload.

    run-pgbench-reuse.py --arm stock|protected --rep N --image POSTGRES_DOM --share PG_SHARE \
        --workload WORKLOAD_DIR --kit KIT --work NEW_DIR [--step-timeout S] [--setting NAME=VALUE]...

WORKLOAD_DIR is a make-pgbench-workload.sh output: the pgbench-initialised cluster, tpcb.sql (the
statements pgbench sent) and native.out (the native single-user replay). The boot is the run-3
PostgreSQL step of eval-plan-p13.txt with tpcb.sql in place of work.sql: trap_platform.py stages
the cluster, share/ and the image, and the guest copies the cluster to /tmp and runs the image as
`postgres --single` with the run-3 options, reading tpcb.sql on its standard input. NEW_DIR then
holds trap_platform.py's results.json, boot-1/ and stage-1/ (the run-3 layout) and rep.json, the
record collect-reuse.py reads: the REUSE-GAP contexts line (parsed with collect.py's pattern),
the exit status, the result rows against native.out, and the hashes of everything staged.
Exits nonzero when the run is to be rejected: incomplete boot, nonzero exit, fault, observer
error, or result rows that differ from the native replay. --setting appends `-c NAME=VALUE` to the
run-3 options, for diagnostic runs outside the pre-registered ones (recorded in rep.json).
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
TRAP_PLATFORM = REPO / 'capstone/runtime/virtual/trap_platform.py'
OPTIONS = ('-c shared_buffers=4MB -c max_connections=10 -c timezone=GMT -c log_timezone=GMT '
           '-c dynamic_shared_memory_type=mmap -c exit_on_error=true')
STEP = ("rm -rf /tmp/pgcluster; cp -a /mnt/vm/pgcluster /tmp/pgcluster && chown -R nobody /tmp/pgcluster && "
        "t0=$(date +%s) && su nobody -s /bin/sh -c 'cd /mnt/vm; ./capstone-vexec ./bin/postgres-{arm}.dom --single "
        "-D /tmp/pgcluster {options} postgres < tpcb.sql'; rc=$?; echo \"PG_TIME:$(( $(date +%s) - t0 ))\"; exit $rc")
ROWS = re.compile(r'\t(?: \d+: [^\n]+|----)')   # collect.py's pg_rows


def load_collect():
    spec = importlib.util.spec_from_file_location('pg_collect', HERE / 'collect.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tree_sha(root):
    h = hashlib.sha256()
    for f in sorted(Path(root).rglob('*')):
        if f.is_file():
            h.update(str(f.relative_to(root)).encode() + b'\0' + bytes.fromhex(sha(f)))
    return h.hexdigest()


def rows_of(text):
    rows = ROWS.findall(text.replace('\r', ''))
    return rows, hashlib.sha256(('\n'.join(rows) + '\n').encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', choices=('stock', 'protected'), required=True)
    p.add_argument('--rep', type=int, required=True)
    for name in ('image', 'share', 'workload', 'kit', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--step-timeout', type=int, default=3600)
    p.add_argument('--setting', action='append', default=[], help='NAME=VALUE, a further -c option')
    a = p.parse_args()
    for setting in a.setting:
        if not re.fullmatch(r'[a-z_]+=[A-Za-z0-9_.]+', setting):
            sys.exit(f'setting {setting!r} is not NAME=VALUE')

    name = f'pg-{a.arm}'
    options = ' '.join([OPTIONS] + [f'-c {setting}' for setting in a.setting])
    command = STEP.format(arm=a.arm, options=options)
    argv = [sys.executable, str(TRAP_PLATFORM), 'run', '--kit', str(a.kit), '--work', str(a.work),
            '--step-timeout', str(a.step_timeout),
            '--file', f'pgcluster={a.workload / "cluster"}', '--file', f'tpcb.sql={a.workload / "tpcb.sql"}',
            '--file', f'share={a.share}', '--file', f'bin/postgres-{a.arm}.dom={a.image}',
            '--step', f'{name}={command}']
    t0 = time.time()
    run = subprocess.run(argv)
    wall = time.time() - t0

    results = json.loads((a.work / 'results.json').read_text())
    r = next(x for x in results if x['name'] == name)
    out = r['output'].replace('\r', '')
    gaps = load_collect().GAP.findall(out)
    reuse_gap = {}
    if len(gaps) == 1:
        g = gaps[0]
        att, iss, rel, reu, dist, cap, err, rr, rs = map(int, g[:9])
        reuse_gap['contexts'] = dict(attempts=att, issues=iss, releases=rel, reuses=reu, distinct_starts=dist,
                                     capacity=cap, error=err, reset_releases=rr, resizes=rs,
                                     bins=list(map(int, g[9].split(','))))
    rows, rows_sha = rows_of(out)
    native_rows, native_sha = rows_of((a.workload / 'native.out').read_text())
    stage = a.work / 'stage-1'
    rec = {
        'program': 'postgres', 'arm': a.arm, 'rep': a.rep,
        'completed': run.returncode == 0 and r['kind'] == 'exit',
        'rc': r['value'] if r['kind'] == 'exit' else None,
        'kind': r['kind'], 'fault': r['fault'], 'kernel_faults': r['kernel_faults'],
        'boot_wall_seconds': round(wall, 1),
        'guest_seconds': int(m.group(1)) if (m := re.search(r'^PG_TIME:(\d+)$', out, re.M)) else None,
        'reuse_gap': reuse_gap, 'reuse_gap_lines': len(gaps),
        'rows': len(rows), 'rows_sha256': rows_sha, 'native_rows': len(native_rows),
        'native_rows_sha256': native_sha, 'rows_equal_native': rows_sha == native_sha,
        'abalance_rows': sum(1 for x in rows if x.startswith('\t 1: abalance = ')),
        'step_command': command, 'extra_settings': a.setting,
        'staged_sha256': {'postgres.dom': sha(stage / f'bin/postgres-{a.arm}.dom'),
                          'capstone-vexec': sha(stage / 'capstone-vexec'), 'gate.sh': sha(stage / 'gate.sh'),
                          'tpcb.sql': sha(stage / 'tpcb.sql'), 'pgcluster_tree': tree_sha(stage / 'pgcluster'),
                          'share_tree': tree_sha(stage / 'share')},
        'workload_manifest': (a.workload / 'manifest.txt').read_text(),
        'platform_sha256': {n: sha(a.kit / 'images' / n) for n in ('Image', 'fw_jump.elf', 'initramfs.cpio.gz')}
                           | {'qemu-system-riscv64': sha(a.kit / 'qemu-system-riscv64'),
                              'capstone-vexec': sha(a.kit / 'adapter/capstone-vexec')},
        'run_staged_sha256': sha(TRAP_PLATFORM),
    }
    (a.work / 'rep.json').write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    rg = reuse_gap.get('contexts')
    print(f'postgres {a.arm} rep{a.rep}: kind={r["kind"]} rc={rec["rc"]} rows={len(rows)} '
          f'rows_equal_native={rec["rows_equal_native"]} error={rg and rg["error"]} '
          f'issues={rg and rg["issues"]} guest={rec["guest_seconds"]}s boot={wall:.0f}s')
    ok = (rec['completed'] and rec['rc'] == 0 and not r['fault'] and not r['kernel_faults'] and rg
          and rg['error'] == 0 and rec['rows_equal_native'])
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
