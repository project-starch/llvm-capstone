#!/usr/bin/env python3
"""Boot the PoisonCap platform once and run its own PoisonCap tests, before any corpus case.

    guest-smoke.py OUT [--platform DIR] [--port PORT]

Boots through ports/common/host/cheribsd/guest.py, the Guest every earlier PoisonCap run used:
single-user, `security.cheri.runtime_revocation_default=0` set on the console before sshd starts
(the published kernel panics, share->excl at vm_map.c:6103, when libc's automatic revocation runs
in ordinary processes; ports/ffmpeg/buffer-pool/results/measurements/20260919-poisoncap-pilot).
Explicit revocation calls are unaffected.

Then, over SSH: the kernel and CHERI sysctls, every cheribsdtest test whose name mentions poison,
and each of those run on its own. OUT/smoke.json holds the result lines; OUT/vm/serial.log the
console. Exit 0 only if the guest booted, the default is 0, at least one poison test exists, and
every one of them passed.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'capstone/ports/common/host/cheribsd'))
from guest import Guest  # noqa: E402

# Generous budgets: the guest runs under TCG on a shared host.
for name, seconds in (('CHERI_GUEST_BOOT_SECONDS', 1200), ('CHERI_GUEST_CMD_SECONDS', 600),
                      ('CHERI_SSH_READY_SECONDS', 600)):
    os.environ.setdefault(name, str(seconds))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', type=Path)
    p.add_argument('--platform', type=Path,
                   default=Path(os.environ.get('POISONCAP_PLATFORM', Path.home() / 'poisoncap-platform')))
    p.add_argument('--port', type=int, default=10087)
    a = p.parse_args()
    work = a.platform / 'work'
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'vm').mkdir()
    guest = Guest(work / 'sdk', work / 'output/rootfs-riscv64-purecap',
                  work / 'output/cheribsd-riscv64-purecap.img', a.out / 'vm', a.port,
                  disable_default_revocation=True)
    (a.out / 'vm/command.json').write_text(json.dumps(guest.argv) + '\n')
    record = {'booted': False}
    try:
        guest.start()
        record['booted'] = True

        def run(cmd, timeout=600):
            r = guest.ssh(cmd, timeout=timeout)
            return {'cmd': cmd, 'rc': r.returncode, 'stdout': r.stdout[-4000:], 'stderr': r.stderr[-2000:]}

        record['uname'] = run('uname -a')
        record['sysctl'] = run('sysctl security.cheri kern.features.cheri_revoke 2>&1')
        tests = None
        for binary in ('cheribsdtest-purecap', 'cheribsdtest-purecap-dynamic'):
            listing = run(f'{binary} -l')
            if listing['rc'] == 0:
                names = re.findall(r'^\s*(\S*poison\S*)', listing['stdout'], re.M | re.I)
                tests = {'binary': binary, 'names': names}
                break
        record['poison_tests'] = tests
        record['runs'] = []
        if tests:
            for name in tests['names']:
                record['runs'].append(run(f"{tests['binary']} {name}", timeout=900))
    finally:
        guest.close()
        (a.out / 'smoke.json').write_text(json.dumps(record, indent=1) + '\n')

    default_off = '.runtime_revocation_default: 0' in record.get('sysctl', {}).get('stdout', '')
    runs = record.get('runs', [])
    ok = record['booted'] and default_off and runs and all(r['rc'] == 0 for r in runs)
    print(f"booted={record['booted']} default_off={default_off} poison_tests={len(runs)} "
          f"passed={sum(r['rc'] == 0 for r in runs)}")
    for r in runs:
        print(f"  rc={r['rc']}  {r['cmd']}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
