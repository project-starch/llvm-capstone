#!/usr/bin/env python3
"""Run one PoisonCap build (build-cases.py) in one boot: controls first, then the cases.

    run-cases.py OUT --build DIR --expect published|spatial [--cases temporal|all|NN,NN]
                 [--libc-revocation off|run] [--controls-only] [--platform DIR] [--port PORT]

The guest boots single-user with security.cheri.runtime_revocation_default=0 before sshd
(platform/poisoncap_guest.py says why). The build's binaries are copied in unchanged and each runs
as its own process, `./NAME TAG`, under a timeout.

--libc-revocation run turns the C library's revocation on for the measured processes only, as the
speedtest1 runs' "on from the run": the command that runs a binary sets the default to 1, execs
`timeout` and the binary under it, and sets it back to 0, so the next ssh session's processes are
exec'd without it (with it on from boot, their kernel fails the copy into the guest). The default,
off, keeps it 0 throughout, as every run before 2026-10-27.

Per process, OUT/results.jsonl gets one line: the exit status and signal, whether BEGIN and RETURNED
were printed, the REPRO_FAULT line (signal, si_code, address, PC offset; repro322_common.h), the
repro lines (heap_perm_poison, lookaside), and the console lines the run added that mention poison
(their QEMU prints "poison exception ..." when it raises the poison fault).

THE GATE. The controls run first and must behave as --expect says, or nothing else runs and the
exit status is 3:
  every control   prints heap_perm_poison=0 (the heap SQLite got cannot bypass poison checks);
  libc_revoke     prints libc_revoke_enabled=1 with --libc-revocation run, 0 otherwise (the
                  process's own malloc_revoke_enabled()); required with run, checked if built;
  reuse_clean     returns, on both arms;
  uaf_read        published: SIGPROT with code 3 (PROT_CHERI_SEALED, the cause their QEMU raises
                  for poison) and a poison line on the console; spatial: returns.
After a passing gate, the evidence controls run (recorded, never gating; controls/uaf_keepperm.c is
the heap with CHERI_PERM_POISON kept), then the cases. `temporal` (default) is every case whose
matrix.tsv axis is temporal.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform'))
import poisoncap_guest  # noqa: E402

CORPUS = poisoncap_guest.REPO / 'capstone/bug-corpora/sqlite/engine-repros'
SIGNALS = {34: 'SIGPROT', 11: 'SIGSEGV', 10: 'SIGBUS', 6: 'SIGABRT'}
REVOCATION_DEFAULT = 'security.cheri.runtime_revocation_default'
CONTROLS = (('reuse_clean', 'ctlreuse'), ('uaf_read', 'ctluaf'))
LIBC_CONTROL = ('libc_revoke', 'ctllibc')
# Recorded after the gate, never gating: what the heap's poison permission does (controls/uaf_keepperm.c),
# what case 15's access pattern does (the next three), and what the C library's layer does (uaf_libc).
EVIDENCE = (('uaf_keepperm', 'ctlkeepperm'), ('uaf_interior', 'ctlinterior'), ('uaf_read_before_free', 'ctlrbf'),
            ('uaf_offsets', 'ctloffsets'), ('uaf_libc', 'ctllibcuaf'))


def temporal_cases():
    axis = {}
    for line in (CORPUS / 'results/matrix.tsv').read_text().splitlines()[1:]:
        cols = line.split('\t')
        if len(cols) > 2:
            axis[cols[1]] = cols[2]
    return {d for d, a in axis.items() if a == 'temporal'}


def select_cases(build, spec):
    """(name, tag) of the cases SPEC names: temporal, all, or comma-separated number prefixes.
    A requested case that the build did not build, a number no case of the build has, and an empty
    selection are errors, never a shorter run: a run over a silent subset reads like a run over all."""
    if spec == 'temporal':
        wanted = temporal_cases()
    elif spec == 'all':
        wanted = set(build['cases'])
    else:
        prefixes = set(spec.split(','))
        unknown = sorted(n for n in prefixes if not any(c[:2] == n for c in build['cases']))
        if unknown:
            sys.exit(f'--cases {spec}: no case of this build has the number {" ".join(unknown)}')
        wanted = {c for c in build['cases'] if c[:2] in prefixes}
    missing = sorted(wanted - set(build['built']))
    if missing:
        sys.exit(f'requested but not built: {" ".join(missing)}')
    if not wanted:
        sys.exit(f'--cases {spec} selects no case')
    return [(c, build['cases'][c]['tag']) for c in sorted(wanted)]


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
        'libc_revoke_enabled': (re.search(r'^repro libc_revoke_enabled=(\d)', stdout, re.M) or [None, None])[1],
        'lookaside_off': bool(re.search(r'^repro lookaside=off rc=0', stdout, re.M)),
        'control_failed': bool(re.search(r'CONTROL-FAILED', stdout)),
        # FAILRC and the cases' own stage reports: `<stage> rc=N`; N != 0 means a step failed.
        'rc_lines': [l.strip() for l in stdout.splitlines() if re.search(r'\brc=[1-9]', l)][:20],
        'tail': stdout.strip().splitlines()[-3:] if stdout.strip() else [],
        'stdout': stdout[-4000000:],
    }


def gate(controls, expect, libc_revocation):
    """The gate's checks on the control rows (name -> results row), by name; the run proceeds only if
    every one holds."""
    checks = {
        'heap_perm_poison=0 in every control': all(r['heap_perm_poison'] == '0' for r in controls.values()),
        'reuse_clean returns': controls['reuse_clean']['returned'] and controls['reuse_clean']['exit'] == 0,
    }
    if 'libc_revoke' in controls:
        want = '1' if libc_revocation == 'run' else '0'
        checks[f'libc_revoke_enabled={want}'] = (controls['libc_revoke']['libc_revoke_enabled'] == want
                                                 and controls['libc_revoke']['returned'])
    u = controls['uaf_read']
    if expect == 'published':
        checks['uaf_read faults with SIGPROT code 3 and a poison console line'] = (
            u['signal'] == 'SIGPROT' and u['fault'] is not None and u['fault']['code'] == '3'
            and bool(u['console_poison']))
    else:
        checks['uaf_read returns'] = u['returned'] and u['exit'] == 0
    return checks


def write_payload(build_dir, names, payload):
    with tarfile.open(payload, 'w:gz') as tar:
        for name in names:
            tar.add(build_dir / name, arcname=name)


class ProcessRunner:
    """Runs one binary per call in the started guest and appends its row to results.jsonl."""

    def __init__(self, guest, serial, results, timeout, libc_revocation):
        self.guest, self.serial, self.results, self.timeout = guest, serial, results, timeout
        self.libc_revocation = libc_revocation
        self.rows = []

    def console_since(self, offset):
        with self.serial.open('rb') as f:
            f.seek(offset)
            return f.read().decode(errors='replace')

    def run(self, kind, name, tag):
        time.sleep(0.5)
        before = self.serial.stat().st_size if self.serial.exists() else 0
        t0 = time.monotonic()
        command = f'timeout {self.timeout} ./{shlex.quote(name)} {shlex.quote(tag)} 2>&1; st=$?'
        if self.libc_revocation == 'run':
            command = (f'sysctl -q {REVOCATION_DEFAULT}=1 > /dev/null; {command}; '
                       f'sysctl -q {REVOCATION_DEFAULT}=0 > /dev/null')
        elif self.libc_revocation != 'off':
            raise ValueError(f'libc_revocation {self.libc_revocation!r}')
        r = self.guest.ssh(f'cd /tmp/allocator-tests && {command}; echo "__EXIT=$st"', timeout=self.timeout + 120)
        seconds = round(time.monotonic() - t0, 1)
        time.sleep(1.0)
        console = self.console_since(before)
        text = r.stdout
        status = re.search(r'__EXIT=(\d+)', text)
        row = {'kind': kind, 'name': name, 'tag': tag, 'seconds': seconds,
               **parse(text.replace(status.group(0), '') if status else text,
                       int(status.group(1)) if status else None),
               'console_poison': [l.strip() for l in console.splitlines() if 'poison' in l.lower()][:12]}
        self.results.write(json.dumps(row) + '\n')
        self.results.flush()
        self.rows.append(row)
        print(f"  {kind:<7} {name:<52} exit={row['exit']} sig={row['signal']} "
              f"fault={row['fault'] and row['fault']['code']} poison_lines={len(row['console_poison'])}")
        return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', type=Path)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--expect', choices=('published', 'spatial'), required=True)
    p.add_argument('--cases', default='temporal')
    p.add_argument('--controls-only', action='store_true')
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--libc-revocation', choices=('off', 'run'), default='off',
                   help="off: the C library's revocation off throughout (as booted); run: on for each "
                        "measured process only")
    poisoncap_guest.add_platform_arguments(p, port=10087)
    a = p.parse_args()
    poisoncap_guest.use_generous_budgets()
    build = json.loads((a.build / 'build.json').read_text())
    cases = select_cases(build, a.cases)
    controls = [(n, t) for n, t in CONTROLS if n in build.get('controls', [])]
    if LIBC_CONTROL[0] in build.get('controls', []):
        controls.append(LIBC_CONTROL)
    elif a.libc_revocation == 'run':
        sys.exit('--libc-revocation run needs the libc_revoke control; rebuild with build-cases.py')
    evidence = [(n, t) for n, t in EVIDENCE if n in build.get('controls', [])]
    if len([n for n, _ in controls if n != LIBC_CONTROL[0]]) != len(CONTROLS):
        sys.exit('the build lacks a control; rebuild with build-cases.py')

    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'vm').mkdir()
    payload = a.out / 'payload.tar.gz'
    write_payload(a.build, [name for name, _ in controls + evidence + cases], payload)
    guest = poisoncap_guest.new_guest(a.platform, a.out / 'vm', a.port)
    meta = {'build': str(a.build), 'expect': a.expect, 'libc_revocation': a.libc_revocation,
            'cases': [c for c, _ in cases],
            'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    gate_ok = False
    results = (a.out / 'results.jsonl').open('w')
    try:
        guest.start()
        guest.copy(payload, 'root@127.0.0.1:/tmp/allocator-tests/payload.tar.gz')
        unpack = guest.ssh('cd /tmp/allocator-tests && tar xzf payload.tar.gz && ls | wc -l')
        meta['unpacked'] = unpack.stdout.strip()
        meta['sysctl'] = guest.ssh('sysctl security.cheri.runtime_revocation_default '
                                   'kern.features.cheri_revoke').stdout.strip()
        runner = ProcessRunner(guest, a.out / 'vm/serial.log', results, a.timeout, a.libc_revocation)
        checks = gate({name: runner.run('control', name, tag) for name, tag in controls}, a.expect,
                      a.libc_revocation)
        meta['gate'] = checks
        gate_ok = all(checks.values())
        print('GATE ' + ('PASS' if gate_ok else 'FAIL') + ': ' + '; '.join(f'{k}={v}' for k, v in checks.items()))
        if gate_ok:
            for name, tag in evidence:
                runner.run('evidence', name, tag)
        if gate_ok and not a.controls_only:
            for name, tag in cases:
                runner.run('case', name, tag)
    finally:
        results.close()
        guest.close()
        meta['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        (a.out / 'run.json').write_text(json.dumps(meta, indent=1) + '\n')
    return 0 if gate_ok else 3


if __name__ == '__main__':
    sys.exit(main())
