#!/usr/bin/env python3
"""Run application-SDK fixtures over 9p + the serial console, and judge them with
the ports' own oracle.

Why this exists. The ports' runner is `capstone-vm` (ffmpeg/app/host/run-qemu.sh is
five lines that say so), and `capstone-vm` provisions the guest over the serial
console and then works over SSH -- it installs its own dropbear from
`--ssh-server`, but this host has no riscv64 dropbear binary to give it, and the
rootfs has none either. This script keeps everything else identical: the guest-side
command is the one `capstone-vm` issues,

    capstone-job <result.json> -- capstone-exec -- <image>.dom

and the verdict comes from `check-safety.py`'s own `classify()`/`matches()`, imported
rather than reimplemented. Only the transport differs.

Three things the earlier ad-hoc scripts got wrong, fixed here:

1. `capstone-job` writes only {version,kind,value}. The FAULT branch of the oracle
   needs a `fault` field, which in `capstone-vm` the HOST adds from the launcher's
   CAPSTONE_FAULT_RECORD file (capstone_vm/cli.py:238-241). Without that merge the
   fault branch is unreachable and every faulting cell reads as an error. We merge it.
2. Each fixture is judged against its OWN section of the serial log, delimited by the
   `=== FIXTURE n BEGIN/EXIT` markers, because the port classifier takes a section.
3. `capstone-vm` forces CAPSTONE_GP_NONLIN=1 and CAPSTONE_REV_NODES=65536
   (cli.py:326-327). We set the same, so a cell here is comparable with one from the
   port's own runner.

Application images need the monitor's process ABI, so the platform must be the pinned
one (its fw_jump has the PROCESS_* ecalls; the installed buildroot monitor has none).
Infrastructure is never a verdict: no capture, no completion marker, or a failed
insmod exits 75.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]          # the repository root: .../capstone/ports/common/application
PORTS = HERE.parents[1]

_spec = importlib.util.spec_from_file_location('check_safety', HERE / 'check-safety.py')
check_safety = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_safety)

PREFIX = {'ffmpeg': 'ffapp', 'wireshark': 'tsapp'}


def read_expectations(expect: Path, arm: str) -> dict[int, list[list[str]]]:
    rows: dict[int, list[list[str]]] = {}
    for line in expect.read_text().splitlines():
        words = line.split()
        if len(words) == 4 and words[0] == arm:
            rows.setdefault(int(words[1]), []).append(words[2:])
    return rows


def section(serial: str, fixture: int) -> str:
    """The fixture's own slice of the console, between its BEGIN and EXIT markers."""
    begin = re.search(rf'^=== FIXTURE {fixture} BEGIN\s*$', serial, re.M)
    if not begin:
        return ''
    end = re.search(rf'^=== FIXTURE {fixture} EXIT .*$', serial[begin.end():], re.M)
    return serial[begin.end():begin.end() + (end.start() if end else len(serial))]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', choices=sorted(PREFIX), required=True)
    p.add_argument('--arm', required=True, help='the arm name as the expect file spells it')
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--expect', type=Path, help="default: the port's safety-expect.txt")
    p.add_argument('--platform', type=Path, default=Path('/tmp/capstone/pinned-platform'),
                   help='holds exec-build/ and the modcapstone module')
    p.add_argument('--qemu-binary', type=Path,
                   default=Path('/tmp/capstone/deleg-gate2/qemu-12/build/qemu-system-riscv64'))
    p.add_argument('--cma', default='1536M')
    p.add_argument('--process-cache-bytes', type=int, default=402653184)
    p.add_argument('--timeout-multiplier', default='4.0',
                   help='scales SETUP timeouts only; the workload gets --seconds-per-fixture')
    p.add_argument('--seconds-per-fixture', type=int, default=180,
                   help='workload budget per fixture. run-domain-smoke.py defaults the guest-command '
                        'timeout to 30 * multiplier, which is a SETUP-sized number: a 9-fixture boot '
                        'blew through it at 240 s with 3 cells done, and the boot is then cut '
                        'mid-stream -- a lost batch that looks like a hang. Set it from the batch size '
                        'instead, as run-domain-smoke.py:450-458 intends.')
    p.add_argument('--verdict', type=Path,
                   help='external verdict script, as check-safety.py --verdict: run as '
                        '<verdict> <out> <expect> <arm> <images> <fixture>... . Required for '
                        "expect files whose vocabulary is not the port's RETURN/FAULT/LEN "
                        '(the pool corpus uses COMPLETE <verdict> / FAULT <lines>)')
    p.add_argument('--expect-override', action='append', default=[],
                   metavar='FIXTURE:KIND:VALUE',
                   help='NEGATIVE CONTROL ONLY: replace a fixture row, to prove the judge '
                        'can report DIFFERS. Never use for a recorded measurement.')
    p.add_argument('fixtures', nargs='+', type=int)
    a = p.parse_args()

    expect = a.expect or PORTS / a.port / 'app/host/safety-expect.txt'
    rows = read_expectations(expect, a.arm)
    for override in a.expect_override:
        number, kind, value = override.split(':')
        rows[int(number)] = [[kind, value]]
        print(f'NEGATIVE CONTROL: fixture {number} row replaced by {kind} {value}', flush=True)
    missing = [n for n in a.fixtures if n not in rows]
    if missing:
        p.error(f'no {a.arm} prediction for fixture(s) {missing} in {expect}')

    # Everything expected to RETURN first: a lost boot then costs the least.
    def faulting(n: int) -> int:
        return int(any(kind in ('FAULT', 'POOLFAIL') for kind, _ in rows[n]))
    order = sorted(a.fixtures, key=lambda n: (faulting(n), n))

    # The canonical module first: a bare */capstone.ko also matches stray copies that
    # earlier runs left in their own output directories.
    module = next(iter(sorted(a.platform.glob('package/modcapstone/module/capstone.ko'))
                       + sorted(a.platform.glob('*/capstone.ko'))), None)
    launcher = a.platform / 'exec-build/capstone-exec'
    helper = a.platform / 'exec-build/capstone-job'
    for required in (module, launcher, helper, a.qemu_binary):
        if required is None or not Path(required).exists():
            print(f'CONTROL-FAILED missing guest artefact: {required}', file=sys.stderr)
            return 75

    share = a.out / 'share'
    share.mkdir(parents=True, exist_ok=False)
    for source in (module, launcher, helper):
        shutil.copy2(source, share / Path(source).name)
    for n in order:
        image = a.images / f'{PREFIX[a.port]}_fx{n}.dom'
        if not image.exists():
            print(f'CONTROL-FAILED missing image {image}', file=sys.stderr)
            return 75
        shutil.copy2(image, share / image.name)

    script = ['#!/bin/sh',
              'cp /mnt/host/capstone-exec /mnt/host/capstone-job /tmp/ '
              '&& chmod +x /tmp/capstone-exec /tmp/capstone-job',
              # run-domain-smoke.py insmods the STOCK module before any guest command,
              # so removing it first is mandatory, not defensive.
              'if [ -c /dev/capstone ]; then rmmod capstone; fi',
              f'insmod /mnt/host/capstone.ko process_cache_bytes={a.process_cache_bytes} '
              '|| { echo INSMOD_FAILED; echo FIXTURES_DONE; exit 0; }']
    for n in order:
        script += [f'echo "=== FIXTURE {n} BEGIN"',
                   f'CAPSTONE_FAULT_RECORD=/mnt/host/fault-{n} /tmp/capstone-job '
                   f'/mnt/host/result-{n}.json -- /tmp/capstone-exec -- '
                   f'/mnt/host/{PREFIX[a.port]}_fx{n}.dom '
                   f'> /mnt/host/out-{n}.txt 2>/mnt/host/err-{n}.txt; '
                   f'echo "=== FIXTURE {n} EXIT $?"']
    # The guest script must exit 0: run_guest_command raises on a non-zero status, and
    # a faulting fixture is a result, not a runner failure.
    script += ['echo FIXTURES_DONE', 'exit 0']
    (share / 'run.sh').write_text('\n'.join(script) + '\n')
    (share / 'run.sh').chmod(0o755)

    serial_path = a.out / 'serial.log'
    environment = dict(os.environ)
    environment['CAPSTONE_GUEST_COMMAND_TIMEOUT'] = str(a.seconds_per_fixture * len(order))
    environment.setdefault('CAPSTONE_GP_NONLIN', '1')
    environment.setdefault('CAPSTONE_REV_NODES', '65536')
    environment.setdefault('CAPSTONE_LLVM_BIN',
                           str(Path(environment.get('CAPSTONE_LLVM_BUILD_DIR', '')) / 'bin'))
    command = [sys.executable, str(REPO / 'capstone/tests/runtime-qemu/run-domain-smoke.py'),
               '--share-dir', str(share.resolve()),
               '--buildroot-dir', str(a.platform / 'images-root'),
               '--qemu-binary', str(a.qemu_binary),
               '--kernel-arg', f'cma={a.cma}',
               '--guest-command', 'sh /mnt/host/run.sh',
               '--success-marker', 'FIXTURES_DONE',
               '--timeout-multiplier', str(a.timeout_multiplier),
               '--log-file', str(serial_path.resolve())]
    with (a.out / 'runner.log').open('w') as log:
        runner = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)

    serial = serial_path.read_text(errors='replace') if serial_path.exists() else ''
    if not serial:
        print(f'NO SERIAL CAPTURE: {a.out}', file=sys.stderr)
        return 75
    if 'INSMOD_FAILED' in serial:
        print(f'INSMOD FAILED, no cell ran: {a.out}', file=sys.stderr)
        return 75
    if 'FIXTURES_DONE' not in serial:
        print(f'BOOT PRODUCED NO RESULT (no completion marker): {a.out}', file=sys.stderr)
        return 75

    # Collect each fixture in the layout check-safety.py produces, so an external
    # verdict script sees exactly what it does under capstone-vm.
    for n in order:
        out_file = share / f'out-{n}.txt'
        (a.out / f'fx{n}.stdout').write_text(
            out_file.read_text(errors='replace') if out_file.exists() else '')
        (a.out / f'fx{n}.qemu').write_text(section(serial, n))
        result_path = share / f'result-{n}.json'
        if result_path.exists():
            record = json.loads(result_path.read_text())
            fault_file = share / f'fault-{n}'
            if fault_file.exists() and fault_file.read_text().strip():
                record['fault'] = fault_file.read_text().strip()
            # ports/common/application/run.py:60-63 adds this in the capstone-vm path, and
            # sublet-port-verdict.py:99 attributes the fault by comparing it with the image's
            # own digest. Omitting it made every faulting cell read "the fault record is for
            # another image (None)" -- an instrument gap that looks like a refutation.
            image = share / f'{PREFIX[a.port]}_fx{n}.dom'
            record['image_sha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
            (a.out / f'fx{n}.json').write_text(json.dumps(record, indent=2) + '\n')

    if a.verdict:
        judge = subprocess.run([sys.executable, str(a.verdict), str(a.out), str(expect),
                                a.arm, str(a.images), *map(str, order)])
        print(f'\nexternal verdict exit {judge.returncode} '
              f'(runner exit {runner.returncode}, which is not the verdict)', flush=True)
        return judge.returncode

    verdicts, failed = [], 0
    for n in order:
        diagnostics = section(serial, n)
        stdout = (share / f'out-{n}.txt').read_text(errors='replace') \
            if (share / f'out-{n}.txt').exists() else ''
        result_path = share / f'result-{n}.json'
        row: dict = {'arm': a.arm, 'fixture': n, 'expected': rows[n]}
        if not result_path.exists():
            row |= {'passed': False, 'error': 'no result record: the fixture never ran'}
            verdicts.append(row)
            failed += 1
            print(f'fx{n}: DIFFERS: no result record', flush=True)
            continue
        result = json.loads(result_path.read_text())
        fault_path = share / f'fault-{n}'
        if fault_path.exists() and fault_path.read_text().strip():
            # Exactly what capstone-vm's host side does; without it the oracle's
            # FAULT branch cannot be reached.
            result['fault'] = fault_path.read_text().strip()
        try:
            (got, detail, explanation), info = check_safety.classify(stdout, diagnostics, result, n)
        except ValueError as error:
            row |= {'passed': False, 'error': str(error), 'result': result}
            verdicts.append(row)
            failed += 1
            print(f'fx{n}: DIFFERS: {error}', flush=True)
            continue
        passed = all(check_safety.matches(got, detail, info, *want, n) for want in rows[n])
        row |= {'passed': passed, 'outcome': got, 'detail': detail,
                'explanation': explanation, 'result': result, **info}
        verdicts.append(row)
        failed += not passed
        print(f'fx{n}: {"AS PREDICTED" if passed else "DIFFERS"}: {got} {detail}', flush=True)

    (a.out / 'verdict.json').write_text(json.dumps(verdicts, indent=2) + '\n')
    print(f'\n{len(order) - failed}/{len(order)} cells as predicted '
          f'(runner exit {runner.returncode}, which is not the verdict)', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
