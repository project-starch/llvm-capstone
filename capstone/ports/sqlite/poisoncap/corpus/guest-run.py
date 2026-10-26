#!/usr/bin/env python3
"""Run one PoisonCap build (build-poisoncap.py) in one boot: controls first, then the cases.

    guest-run.py OUT --build DIR --expect published|spatial [--cases temporal|all|NN,NN]
                 [--controls-only] [--platform DIR] [--port PORT]

The guest boots through ports/common/host/cheribsd/guest.py, as platform/smoke-test.py does (single-user,
security.cheri.runtime_revocation_default=0 before sshd; the reason is in that script). The build's
binaries are copied in unchanged and each runs as its own process, `./NAME TAG`, under a timeout.

Per process, OUT/results.jsonl gets one line: the exit status and signal, whether BEGIN and RETURNED
were printed, the REPRO_FAULT line (signal, si_code, address, PC offset; repro322_common.h), the
repro lines (heap_perm_poison, lookaside), and the console lines the run added that mention poison
(their QEMU prints "poison exception ..." when it raises the poison fault).

THE GATE. The controls run first and must behave as --expect says, or nothing else runs and the
exit status is 3:
  every control   prints heap_perm_poison=0 (the heap SQLite got cannot bypass poison checks);
  reuse_clean     returns, on both arms;
  uaf_read        published: SIGPROT with code 3 (PROT_CHERI_SEALED, the cause their QEMU raises
                  for poison) and a poison line on the console; spatial: returns.
After a passing gate, controls/uaf_keepperm.c runs as recorded evidence (the heap with
CHERI_PERM_POISON kept), then the cases. `temporal` (default) is every case whose matrix.tsv axis
is temporal.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time

REPO = Path(__file__).resolve().parents[5]
CORPUS = REPO / 'capstone/bug-corpora/sqlite/engine-repros'
sys.path.insert(0, str(REPO / 'capstone/ports/common/host/cheribsd'))
from guest import Guest  # noqa: E402

for name, seconds in (('CHERI_GUEST_BOOT_SECONDS', 1200), ('CHERI_GUEST_CMD_SECONDS', 600),
                      ('CHERI_SSH_READY_SECONDS', 600)):
    os.environ.setdefault(name, str(seconds))
SIGNALS = {34: 'SIGPROT', 11: 'SIGSEGV', 10: 'SIGBUS', 6: 'SIGABRT'}
CONTROLS = (('reuse_clean', 'ctlreuse'), ('uaf_read', 'ctluaf'))
# Recorded after the gate, never gating: what the heap's poison permission does (controls/uaf_keepperm.c).
EVIDENCE = (('uaf_keepperm', 'ctlkeepperm'), ('uaf_interior', 'ctlinterior'), ('uaf_read_before_free', 'ctlrbf'), ('uaf_offsets', 'ctloffsets'))


def temporal_cases():
    axis = {}
    for line in (CORPUS / 'results/matrix.tsv').read_text().splitlines()[1:]:
        cols = line.split('\t')
        if len(cols) > 2:
            axis[cols[1]] = cols[2]
    return {d for d, a in axis.items() if a == 'temporal'}


def parse(stdout, exit_status):
    fault = re.search(r'^REPRO_FAULT sig=(\d+) code=(-?\d+) addr=(\S+) pc_off=(\S+) obj=(\S+)', stdout, re.M)
    sig = exit_status - 128 if exit_status is not None and exit_status > 128 else None
    return {
        'exit': exit_status,
        'signal': SIGNALS.get(sig, sig) if sig else None,
        'begin': bool(re.search(r' BEGIN$', stdout, re.M)),
        'returned': bool(re.search(r' RETURNED$', stdout, re.M)),
        'fault': dict(zip(('sig', 'code', 'addr', 'pc_off', 'obj'), fault.groups())) if fault else None,
        'heap_perm_poison': (re.search(r'^repro heap_perm_poison=(\d)', stdout, re.M) or [None, None])[1],
        'lookaside_off': bool(re.search(r'^repro lookaside=off rc=0', stdout, re.M)),
        'control_failed': bool(re.search(r'CONTROL-FAILED', stdout)),
        # FAILRC and the cases' own stage reports: `<stage> rc=N`; N != 0 means a step failed.
        'rc_lines': [l.strip() for l in stdout.splitlines() if re.search(r'\brc=[1-9]', l)][:20],
        'tail': stdout.strip().splitlines()[-3:] if stdout.strip() else [],
        'stdout': stdout[-4000000:],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', type=Path)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--expect', choices=('published', 'spatial'), required=True)
    p.add_argument('--cases', default='temporal')
    p.add_argument('--controls-only', action='store_true')
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--platform', type=Path,
                   default=Path(os.environ.get('POISONCAP_PLATFORM', Path.home() / 'poisoncap-platform')))
    p.add_argument('--port', type=int, default=10087)
    a = p.parse_args()
    build = json.loads((a.build / 'build.json').read_text())
    if a.cases == 'temporal':
        wanted = temporal_cases()
    elif a.cases == 'all':
        wanted = set(build['cases'])
    else:
        prefixes = set(a.cases.split(','))
        wanted = {c for c in build['cases'] if c[:2] in prefixes}
    cases = [(c, build['cases'][c]['tag']) for c in sorted(build['cases']) if c in wanted and c in build['built']]
    controls = [(n, t) for n, t in CONTROLS if n in build.get('controls', [])]
    evidence = [(n, t) for n, t in EVIDENCE if n in build.get('controls', [])]
    if len(controls) != len(CONTROLS):
        sys.exit('the build lacks a control; rebuild with build-poisoncap.py')

    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'vm').mkdir()
    payload = a.out / 'payload.tar.gz'
    with tarfile.open(payload, 'w:gz') as tar:
        for name, _ in controls + evidence + cases:
            tar.add(a.build / name, arcname=name)
    work = a.platform / 'work'
    guest = Guest(work / 'sdk', work / 'output/rootfs-riscv64-purecap',
                  work / 'output/cheribsd-riscv64-purecap.img', a.out / 'vm', a.port,
                  disable_default_revocation=True)
    (a.out / 'vm/command.json').write_text(json.dumps(guest.argv) + '\n')
    serial = a.out / 'vm/serial.log'
    meta = {'build': str(a.build), 'expect': a.expect, 'cases': [c for c, _ in cases],
            'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    rows, gate_ok = [], False
    results = (a.out / 'results.jsonl').open('w')
    try:
        guest.start()
        guest.copy(payload, 'root@127.0.0.1:/tmp/allocator-tests/payload.tar.gz')
        unpack = guest.ssh('cd /tmp/allocator-tests && tar xzf payload.tar.gz && ls | wc -l')
        meta['unpacked'] = unpack.stdout.strip()
        meta['sysctl'] = guest.ssh('sysctl security.cheri.runtime_revocation_default '
                                   'kern.features.cheri_revoke').stdout.strip()

        def run_one(kind, name, tag):
            time.sleep(0.5)
            before = serial.stat().st_size if serial.exists() else 0
            t0 = time.monotonic()
            r = guest.ssh(f'cd /tmp/allocator-tests && timeout {a.timeout} ./{shlex.quote(name)} '
                          f'{shlex.quote(tag)} 2>&1; echo "__EXIT=$?"', timeout=a.timeout + 120)
            seconds = round(time.monotonic() - t0, 1)
            time.sleep(1.0)
            with serial.open('rb') as f:
                f.seek(before)
                console = f.read().decode(errors='replace')
            text = r.stdout
            status = re.search(r'__EXIT=(\d+)', text)
            row = {'kind': kind, 'name': name, 'tag': tag, 'seconds': seconds,
                   **parse(text.replace(status.group(0), '') if status else text,
                           int(status.group(1)) if status else None),
                   'console_poison': [l.strip() for l in console.splitlines() if 'poison' in l.lower()][:12]}
            results.write(json.dumps(row) + '\n')
            results.flush()
            rows.append(row)
            print(f"  {kind:<7} {name:<52} exit={row['exit']} sig={row['signal']} "
                  f"fault={row['fault'] and row['fault']['code']} poison_lines={len(row['console_poison'])}")
            return row

        ctl = {name: run_one('control', name, tag) for name, tag in controls}
        checks = {
            'heap_perm_poison=0 in every control': all(r['heap_perm_poison'] == '0' for r in ctl.values()),
            'reuse_clean returns': ctl['reuse_clean']['returned'] and ctl['reuse_clean']['exit'] == 0,
        }
        if a.expect == 'published':
            u = ctl['uaf_read']
            checks['uaf_read faults with SIGPROT code 3 and a poison console line'] = (
                u['signal'] == 'SIGPROT' and u['fault'] is not None and u['fault']['code'] == '3'
                and bool(u['console_poison']))
        else:
            checks['uaf_read returns'] = ctl['uaf_read']['returned'] and ctl['uaf_read']['exit'] == 0
        meta['gate'] = checks
        gate_ok = all(checks.values())
        print('GATE ' + ('PASS' if gate_ok else 'FAIL') + ': ' + '; '.join(f'{k}={v}' for k, v in checks.items()))
        if gate_ok:
            for name, tag in evidence:
                run_one('evidence', name, tag)
        if gate_ok and not a.controls_only:
            for name, tag in cases:
                run_one('case', name, tag)
    finally:
        results.close()
        guest.close()
        meta['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        (a.out / 'run.json').write_text(json.dumps(meta, indent=1) + '\n')
    return 0 if gate_ok else 3


if __name__ == '__main__':
    sys.exit(main())
