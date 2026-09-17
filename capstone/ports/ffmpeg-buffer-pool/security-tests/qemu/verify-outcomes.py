#!/usr/bin/env python3
"""Paired pool lifetime tests; fault verdicts require stage, cause, and exact PC."""
import argparse
import json
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys

NAMES = ['shared-references-callbacks-deferred-close', 'buffer-read-after-return',
         'buffer-write-after-return', 'buffer-read-after-reuse', 'buffer-write-after-reuse',
         'refstruct-read-after-return', 'refstruct-write-after-reuse', 'refstruct-stale-unref',
         'nested-child-read-after-reset', 'buffer-read-after-backing-free',
         'refstruct-metadata-underflow', 'buffer-one-past-end',
         'retained-stale-reference-long-run', 'valid-reference-long-run']
parser = argparse.ArgumentParser()
parser.add_argument('output', type=pathlib.Path)
parser.add_argument('--cases', default='0,1,2,3,4,5,6,7,8,9,10,11')
parser.add_argument('--modes', default='0,1,2', help='Capstone modes: 0=bounds, 1=backing lifetime, 2=Sublet pool leases')
parser.add_argument('--rounds', type=int, default=70000)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
work = pathlib.Path(os.environ['FFPOOL_WORK'])
repo = pathlib.Path(os.environ['CAPSTONE_REPO_ROOT'])
binary = work / 'combined-port-capstone'
verdicts = []
for mode in map(int, args.modes.split(',')):
    for case in map(int, args.cases.split(',')):
        if mode not in (0, 1, 2) or not 0 <= case < len(NAMES):
            raise SystemExit('invalid mode or case')
        run = args.output / f'mode-{mode}-case-{case}'
        run.mkdir(exist_ok=False)
        share = run / 'share'
        share.mkdir()
        shutil.copy2(binary / 'security.dom', share / 'security.dom')
        shutil.copy2(binary / 'host.user', share / 'host.user')
        header = [0x4650465452433032, 1] + [0] * 10 + [case, args.rounds, 0, 0]
        (share / 'trace.bin').write_bytes(struct.pack('<16Q', *header) + bytes(128))
        status = 304 if mode == 2 and case == 7 else 0
        fault = case in (10, 11) or (case == 9 and mode >= 1) or (mode == 2 and case in (1, 2, 3, 4, 5, 6, 8, 12))
        guest = f'''#!/bin/sh
set -e
cp /mnt/host/security.dom /tmp/security.dom
cp /mnt/host/host.user /tmp/security-host
cp /mnt/host/trace.bin /tmp/security-trace.bin
status=0
/tmp/security-host /tmp/security.dom /tmp/security-trace.bin /tmp/security-result.bin {mode} > /tmp/security.out || status=$?
cat /tmp/security.out
cp /tmp/security-result.bin /mnt/host/result.bin
test "$status" = {1 if status else 0}
grep -q '^FF2 return={status if status else 42044} status={status} events=' /tmp/security.out
echo FF2_SECURITY_DONE
'''
        (share / 'run.sh').write_text(guest)
        command = [sys.executable, str(repo / 'capstone/tests/runtime-qemu/run-domain-smoke.py'),
                   '--share-dir', str(share.resolve()), '--log-file', str((run / 'serial.log').resolve()),
                   '--kernel-arg', 'cma=512M', '--timeout-multiplier', '12',
                   '--guest-command', 'sh /mnt/host/run.sh', '--success-marker', 'FF2_SECURITY_DONE']
        with (run / 'runner.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        serial = (run / 'serial.log').read_text(errors='replace')
        faults = re.findall(r'domain halted by capability fault: cause = (\d+), pc = (0x[0-9a-f]+)', serial)
        stage = f'Print = Scalar(0x{0xff25000000000000 | case:x})'
        row = dict(mode=mode, case=case, name=NAMES[case], expected='fault' if fault else ('rejected' if status else 'completed'),
                   runner_exit=result.returncode, node_budget=int(os.environ['CAPSTONE_REV_NODES']), rounds=args.rounds,
                   log=str(run / 'serial.log'))
        if fault:
            following = serial.split(stage, 1)[-1] if stage in serial else ''
            sites = re.findall(r'Print = Cap\(\d+, 0x[0-9a-f]+, (0x[0-9a-f]+),', following)
            site = 1 if case in (2, 4, 6) else 0
            cause, pc = faults[-1] if faults else ('0', '0')
            # This pinned QEMU reports bounds failures as load-access fault 5,
            # with the explicit OOB diagnostic and offending range.
            allowed = (5,) if case in (10, 11) else (24, 25)
            row.update(cause=int(cause), pc=pc, expected_pc=sites[site] if len(sites) > site else None)
            row['passed'] = (stage in serial and len(sites) > site and int(cause) in allowed and
                             int(pc, 16) == int(sites[site], 16) and 'FF2 return=42044' not in serial)
            if case in (10, 11):
                row['passed'] &= 'Cap mem access OOB:' in following
        else:
            row['passed'] = result.returncode == 0 and not faults and stage in serial
            if status:
                raw = (share / 'result.bin').read_bytes()
                row['passed'] &= struct.unpack_from('<16Q', raw)[2] == status
        verdicts.append(row)
        (args.output / 'verdicts.json').write_text(json.dumps(verdicts, indent=2) + '\n')
        print(f"{'PASS' if row['passed'] else 'FAIL'} mode={mode} {NAMES[case]} expected={row['expected']}", flush=True)
        if not row['passed']:
            raise SystemExit(1)
