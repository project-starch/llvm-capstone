#!/usr/bin/env python3
"""Qualify the launcher's mapping pool (CAPSTONE_VM_MAPCACHE, exec.c; module CV_PARK / CV_REVIVE).

mapcache-contract.c runs once per case, with the pool off and on. What the pool must not change:
the second block of a size is zero data and no capability tag (reuse), a pointer to a freed block
faults with cause 25 whether or not its backing was given out again (stale), the program's output
is the same (churn, threads), a block whose protections were changed is not pooled (protect), the
pool stays under its cap (bounded), lifetime records are recycled without end (nodes), the pool is given
back at the first ENOMEM (pressure).

Negative controls run the same cases against a broken variant and must FAIL a named check:
  --module-args test_revive_skip_clear=1       (hooks module) reuse without clearing
  --module-args test_revive_keep_ancestor=1    (hooks module) reuse without retiring the old lifetime
  a launcher built with -DCAPSTONE_MAPCACHE_MUTANT_UNBOUNDED / _PROT / _NOFLUSH
--expect-fail NAME... lists the checks a control must fail; the exit status is then 0 only when every
listed check failed and no other check did.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import memory_mode

HERE = Path(__file__).resolve().parent
CAP = 262144
POOL = '1048576'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block(log, name):
    rows = re.findall(r'MC_CASE:' + re.escape(name) + r'\n(.*?)MC_EXIT:' + re.escape(name) + r':(\d+)\n', log, re.S)
    return (rows[0][0], int(rows[0][1])) if len(rows) == 1 else (None, None)


def stats(body):
    line = re.search(r'^CAPSTONE_VM_MAPCACHE (.*)$', body, re.M)
    return {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line[1])} if line else None


def vmstats(body):
    line = re.search(r'^CAPSTONE_VM_STATS (.*)$', body, re.M)
    return {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line[1])} if line else None


def causes(body):
    return [int(n) for n in re.findall(r'domain fault cause=(\d+)', body)]


def grade(log, nodes_cycles):
    t = {}
    b = {n: block(log, n) for n in CASES}
    def ok(name, marker):
        body, rc = b[name]
        return body is not None and rc == 0 and marker in body and 'MAPCACHE_FAIL' not in body and 'domain fault' not in body
    t['reuse_off'] = ok('reuse-off', 'MAPCACHE_REUSE_OK') and stats(b['reuse-off'][0]) is None
    body = b['reuse-on'][0] or ''
    s = stats(body)
    t['reuse_on_clean'] = ok('reuse-on', 'MAPCACHE_REUSE_OK same_address=1') and s is not None and s['hits'] >= 1
    # The stale cases: the fault must come AFTER the setup marker, and be the only fault.
    for name, marker, trace in (('stale-off', 'MAPCACHE_STALE_REUSED', None), ('stale-on', 'MAPCACHE_STALE_REUSED same_address=1', 'hit'),
                                ('stale-pooled-on', 'MAPCACHE_STALE_FREED', 'park'),
                                ('stale-grant-off', 'MAPCACHE_STALE_GRANT_REUSED', None),
                                ('stale-grant-on', 'MAPCACHE_STALE_GRANT_REUSED same_address=1', 'hit'),
                                ('stale-grant-pooled-on', 'MAPCACHE_STALE_GRANT_FREED', 'park')):
        body, rc = b[name]
        good = body is not None and rc == 139 and marker in body and causes(body) == [25] and 'MAPCACHE_FAIL' not in body
        if trace == 'hit':
            good = good and len(re.findall(r'CAPSTONE_VM_MAPCACHE_HIT ', body)) >= 1
        if trace == 'park':
            good = good and len(re.findall(r'CAPSTONE_VM_MAPCACHE_PARK ', body)) >= 1 and 'CAPSTONE_VM_MAPCACHE_HIT ' not in body
        t['stale_' + name[6:].replace('-', '_')] = good  # stale_off, stale_on, stale_pooled_on, stale_grant_*
    off, on = b['churn-off'][0] or '', b['churn-on'][0] or ''
    line = lambda x: re.findall(r'^MAPCACHE_CHURN_OK .*$', x, re.M)
    s = stats(on)
    t['churn_off'] = ok('churn-off', 'MAPCACHE_CHURN_OK')
    t['churn_on_identical'] = ok('churn-on', 'MAPCACHE_CHURN_OK') and line(off) == line(on) and len(line(on)) == 1 and s is not None and s['hits'] >= 1
    s = stats(b['bounded-on'][0] or '')
    t['bounded'] = ok('bounded-on', 'MAPCACHE_BOUNDED_DONE') and s is not None and s['parks'] >= 1 and s['evictions'] >= 1 \
        and s['pooled_bytes'] <= s['cap'] == CAP
    t['protect_off'] = ok('protect-off', 'MAPCACHE_PROTECT_OK')
    t['protect_on'] = ok('protect-on', 'MAPCACHE_PROTECT_OK')
    off, on = b['threads-off'][0] or '', b['threads-on'][0] or ''
    line = lambda x: re.findall(r'^MAPCACHE_THREADS_OK .*$', x, re.M)
    s = stats(on)
    t['threads_off'] = ok('threads-off', 'MAPCACHE_THREADS_OK')
    t['threads_on_identical'] = ok('threads-on', 'MAPCACHE_THREADS_OK') and line(off) == line(on) and len(line(on)) == 1 and s is not None and s['hits'] >= 1
    body = b['nodes-on'][0] or ''
    s, v = stats(body), vmstats(body)
    t['nodes_recycled'] = ok('nodes-on', 'MAPCACHE_NODES_OK cycles=%d ' % nodes_cycles) and s is not None and v is not None \
        and s['hits'] >= nodes_cycles - 8 and v['collections'] >= 1
    s = stats(b['pressure-on'][0] or '')
    t['pressure_flush'] = ok('pressure-on', 'MAPCACHE_PRESSURE_OK') and s is not None and s['flushes'] >= 1 and s['evictions'] >= 1
    return t


CASES = ['reuse-off', 'reuse-on', 'stale-off', 'stale-on', 'stale-pooled-on', 'stale-grant-off', 'stale-grant-on',
         'stale-grant-pooled-on', 'churn-off', 'churn-on', 'bounded-on',
         'protect-off', 'protect-on', 'threads-off', 'threads-on', 'nodes-on', 'pressure-on']
# case -> (mode, environment)
SPEC = {
    'reuse-off': ('reuse', 'CAPSTONE_VM_STATS=1'),
    'reuse-on': ('reuse', f'CAPSTONE_VM_STATS=1 CAPSTONE_VM_MAPCACHE={POOL}'),
    'stale-off': ('stale', 'CAPSTONE_EXEC_DIAGNOSTICS=1'),
    'stale-on': ('stale', f'CAPSTONE_VM_TRACE=1 CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_MAPCACHE={POOL}'),
    'stale-pooled-on': ('stale-pooled', f'CAPSTONE_VM_TRACE=1 CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_MAPCACHE={POOL}'),
    'stale-grant-off': ('stale-grant', 'CAPSTONE_EXEC_DIAGNOSTICS=1'),
    'stale-grant-on': ('stale-grant', f'CAPSTONE_VM_TRACE=1 CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_MAPCACHE={POOL}'),
    'stale-grant-pooled-on': ('stale-grant-pooled', f'CAPSTONE_VM_TRACE=1 CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_MAPCACHE={POOL}'),
    'churn-off': ('churn', 'CAPSTONE_VM_STATS=1'),
    'churn-on': ('churn', f'CAPSTONE_VM_STATS=1 CAPSTONE_VM_MAPCACHE={POOL}'),
    'bounded-on': ('bounded', f'CAPSTONE_VM_STATS=1 CAPSTONE_VM_MAPCACHE={CAP}'),
    'protect-off': ('protect', 'CAPSTONE_VM_STATS=1'),
    'protect-on': ('protect', f'CAPSTONE_VM_STATS=1 CAPSTONE_VM_MAPCACHE={POOL}'),
    'threads-off': ('threads', 'CAPSTONE_VM_STATS=1'),
    'threads-on': ('threads', f'CAPSTONE_VM_STATS=1 CAPSTONE_VM_MAPCACHE={POOL}'),
    'nodes-on': ('nodes', f'CAPSTONE_VM_STATS=1 CAPSTONE_VM_MAPCACHE={POOL}'),
    'pressure-on': ('pressure', f'CAPSTONE_VM_STATS=1 CAPSTONE_VM_MAPCACHE={POOL}'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('adapter', 'application', 'qemu', 'images', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--module', type=Path, help='a module to load instead of ADAPTER/module/capstone_vm.ko (the hooks build)')
    p.add_argument('--module-args', default='')
    p.add_argument('--nodes-cycles', type=int, default=12000)
    p.add_argument('--timeout', type=int, default=1500)
    p.add_argument('--expect-fail', nargs='*', default=[])
    p.add_argument('--exact-bounds', action=argparse.BooleanOptionalAction, default=False)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    module = a.module or a.adapter / 'module/capstone_vm.ko'
    inputs = {'capstone-vexec': a.adapter / 'capstone-vexec', 'capstone_vm.ko': module, 'mapcache.dom': a.application}
    for name, path in inputs.items():
        shutil.copy2(path, stage / name)
    gate = ['#!/bin/sh', 'cd /mnt/vm || exit 1', 'dmesg -n 1',
            f'insmod capstone_vm.ko {a.module_args} || exit 1', memory_mode.echo('MC_'),
            f'export MAPCACHE_CYCLES={a.nodes_cycles}']
    for name in CASES:
        mode, env = SPEC[name]
        gate += [f'echo MC_CASE:{name}', f'env {env} ./capstone-vexec mapcache.dom {mode} 2>&1', f'echo MC_EXIT:{name}:$?']
    gate += ['echo MC_DMESG_WARN:$(dmesg | grep -c -i -E "warning|bug:|oops")', 'rmmod capstone_vm', 'echo MC_CLEANUP:$?',
             'echo VIRTUAL_STAGED_DONE']
    (stage / 'gate.sh').write_text('\n'.join(gate) + '\n')
    run = subprocess.run([sys.executable, str(HERE / 'run-staged.py'), '--qemu', str(a.qemu.resolve()), '--images', str(a.images.resolve()),
                          '--stage', str(stage.resolve()), '--work', str((a.work / 'guest').resolve()), '--timeout', str(a.timeout),
                          '--exact-bounds' if a.exact_bounds else '--no-exact-bounds'])
    log = (a.work / 'guest/serial.log').read_text(errors='replace').replace('\r', '')
    checks = grade(log, a.nodes_cycles)
    checks['cleanup'] = run.returncode == 0 and 'MC_CLEANUP:0\n' in log
    checks['no_kernel_warning'] = 'MC_DMESG_WARN:0\n' in log and 'Kernel panic' not in log
    checks['memory_mode'] = memory_mode.verified(log, False, 1)
    failed = sorted(k for k, v in checks.items() if not v)
    if a.expect_fail:
        good = failed == sorted(a.expect_fail)
        status = 'CONTROL_FIRED' if good else 'CONTROL_WRONG'
    else:
        good = not failed
        status = 'PASS' if good else 'FAIL'
    result = dict(status=status, tests=checks, failed=failed, expect_fail=a.expect_fail, module_args=a.module_args,
                  input_sha256={n: sha(pth) for n, pth in inputs.items()}, runner_sha256=sha(Path(__file__)))
    (a.work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, ok in checks.items():
        print(('PASS ' if ok else 'FAIL ') + name)
    print(status)
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
