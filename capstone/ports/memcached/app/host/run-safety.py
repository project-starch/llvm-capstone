#!/usr/bin/env python3
"""The memcached Safety milestone: run fixtures in the safety image of one heap arm, one guest boot,
and judge each against host/safety-expect.txt.

    run-safety.py --arm level0|shrink|sublet --out DIR [--expect FILE] FIXTURE...

Env: MC_WORK (host/build-safety.sh's images), CAPSTONE_VM_UP_ARGS (capstone-vm up platform arguments),
CAPSTONE_BUILDROOT_DIR (the guest cross compiler), capstone-vm on PATH.

Each fixture is one server process: mc-harness --fixture N starts memcached-safety-<arm>.dom under
capstone-job as `nobody`, waits until it listens, and sends the hidden `mc_capstone_fixture N` (patch
0005), which runs fixture N on the worker that took the connection. The fixture either returns (it
prints its mark and the process exits with it) or faults (capstone-job records the signal, the
launcher writes the fault record). Each fixture runs in its own `capstone-vm exec`, and the slice of
qemu.log written during it is that fixture's diagnostics.

The judgement is ports/common/application/check-safety.py's, imported, not copied: its classify()
(the fixture's own mark corroborated by the exit status, or a domain SIGSEGV whose fault-record pc
matches a QEMU diagnostic, attributed only after the fixture's touch line and at its printed target)
and its matches() against the prediction. The fixture prints MCAPP-FIX lines; they are handed over
under the FFAPP- prefix that classifier reads, which is the only translation.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

APP = Path(__file__).resolve().parents[1]
PORTS = APP.parents[1]
spec = importlib.util.spec_from_file_location('check_safety', PORTS / 'common/application/check-safety.py')
check_safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_safety)

PORT = 21299
FLAGS = ['-l', '127.0.0.1', '-p', str(PORT), '-U', '0', '-m', '64', '-t', '4']


def vm(state, *args, **kw):
    return subprocess.run(['capstone-vm', '--state', str(state), *args], **kw)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arm', choices=['level0', 'shrink', 'sublet'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expect', type=Path, default=APP / 'host/safety-expect.txt')
    parser.add_argument('fixtures', nargs='+', type=int)
    args = parser.parse_args()

    expectations = {}
    for line in args.expect.read_text().splitlines():
        words = line.split('#', 1)[0].split()
        if len(words) == 4 and words[0] == args.arm:
            expectations.setdefault(int(words[1]), []).append(words[2:])
    for number in args.fixtures:
        if number not in expectations:
            parser.error(f'no prediction for {args.arm} fixture {number}')

    work = Path(os.environ['MC_WORK'])
    image = work / 'safety' / f'memcached-safety-{args.arm}.dom'
    if not image.is_file():
        parser.error(f'no {image}: run host/build-safety.sh')
    args.out.mkdir(parents=True, exist_ok=False)
    share = args.out / 'share'
    share.mkdir()
    shutil.copy2(image, share / 'memcached-safety.dom')
    xcc = Path(os.environ['CAPSTONE_BUILDROOT_DIR']) / 'build/host/bin/riscv64-buildroot-linux-gnu-gcc'
    subprocess.run([str(xcc), '-O1', '-Wall', '-o', str(share / 'mc-harness'),
                    str(APP / 'host/mc-harness/mc-harness.c'), '-lpthread'], check=True)
    inputs = {name: subprocess.run(['sha256sum', str(share / name)], capture_output=True, text=True,
                                   check=True).stdout.split()[0] for name in ('memcached-safety.dom', 'mc-harness')}
    (args.out / 'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    print(f'safety {args.arm}: image {inputs["memcached-safety.dom"][:16]} harness {inputs["mc-harness"][:16]}', flush=True)

    state = args.out / 'vm'
    for _ in range(2400):
        shutil.rmtree(state, ignore_errors=True)
        up = vm(state, 'up', *shlex.split(os.environ['CAPSTONE_VM_UP_ARGS']), '--share', str(share),
                '--boot-timeout', '600', capture_output=True, text=True)
        (args.out / 'up.log').write_text(up.stdout + up.stderr)
        if up.returncode == 0:
            break
        if 'Another Capstone VM owns' not in up.stdout + up.stderr:
            print(up.stderr[-2000:], file=sys.stderr)
            return 4
        time.sleep(3)
    results = []
    try:
        guest = vm(state, 'exec', 'sh', '-c', 'sha256sum /usr/bin/capstone-exec /usr/bin/capstone-job',
                   capture_output=True, text=True)
        (args.out / 'guest-tools.txt').write_text(guest.stdout)
        print(guest.stdout, end='', flush=True)
        qemu_log = state / 'qemu.log'
        for number in args.fixtures:
            start = qemu_log.stat().st_size if qemu_log.exists() else 0
            script = (f'rm -rf /tmp/fx /tmp/fx-fault.txt; mkdir -p /tmp/fx && '
                      f'CAPSTONE_FAULT_RECORD=/tmp/fx-fault.txt /mnt/host/mc-harness --out /tmp/fx --port {PORT} '
                      f'--fixture {number} -- /usr/bin/capstone-job /tmp/fx/job.json --user 65534:65534 -- '
                      f'/usr/bin/capstone-exec /mnt/host/memcached-safety.dom {" ".join(FLAGS)}; '
                      f'echo harness rc=$?; if [ -s /tmp/fx-fault.txt ]; then cp /tmp/fx-fault.txt /tmp/fx/fault.txt; fi; '
                      f'rm -rf /mnt/host/fx{number}; cp -r /tmp/fx /mnt/host/fx{number}; true')
            run = vm(state, 'exec', 'sh', '-c', script, capture_output=True, text=True)
            time.sleep(1)   # let QEMU's diagnostics reach its log
            with qemu_log.open('rb') as stream:
                stream.seek(start)
                diagnostics = stream.read().decode(errors='replace')
            d = share / f'fx{number}'
            (args.out / f'fx{number}.exec').write_text(run.stdout + run.stderr)
            (args.out / f'fx{number}.qemu').write_text(diagnostics)
            stdout = (d / 'server.out').read_text(errors='replace') if (d / 'server.out').exists() else ''
            job = json.loads((d / 'job.json').read_text()) if (d / 'job.json').exists() else {}
            result = {'kind': job.get('kind'), 'value': job.get('value')}
            if (d / 'fault.txt').exists():
                result['fault'] = (d / 'fault.txt').read_text().strip()
            try:
                (got, detail, explanation), info = check_safety.classify(
                    stdout.replace('MCAPP-', 'FFAPP-'), diagnostics, result, number)
            except ValueError as error:
                got, detail, explanation, info = 'ERROR', str(error), '', {'len': None}
            passed = got != 'ERROR' and all(
                check_safety.matches(got, detail, info, *want, number) for want in expectations[number])
            results.append(dict(fixture=number, passed=passed, outcome=got, detail=detail, explanation=explanation,
                                expected=expectations[number], result=result, **info))
            print(f'fx{number}: {"AS PREDICTED" if passed else "DIFFERS"}: {got} {detail}'
                  f'  (predicted {"; ".join(" ".join(w) for w in expectations[number])};'
                  f' {result["kind"]} {result["value"]}, len={info.get("len")})', flush=True)
    finally:
        vm(state, 'down', capture_output=True)
    (args.out / 'verdict.json').write_text(json.dumps(results, indent=2) + '\n')
    return 0 if results and all(row['passed'] for row in results) else 1


if __name__ == '__main__':
    sys.exit(main())
