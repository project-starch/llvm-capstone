#!/usr/bin/env python3
"""Check a fast-sweep module against a real tagged-page omission control.

Run through the host's gate queue. --probes supplies namespace-check and
namespace-probe.ko from the namespace fixture; --application is node-contract.c
built with the matching SDK. --mutant is a separately built module that skips
tagged pages in next_tagged; --shared-mutant omits the mapcount checks. The
probes directory also supplies sweep-shared built against namespace-probe.h.
Production and mutants use identical assertions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
SCRIPT = '''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod namespace-probe.ko || exit 1
for module in good omit; do
  insmod $module.ko unpinned=1 unpinned_fast_sweep=1 supervisor_slots=32 || exit 1
  echo SWEEP_MODULE:$module
  echo SWEEP_MODE:$(cat /sys/module/capstone_vm/parameters/unpinned):$(cat /sys/module/capstone_vm/parameters/unpinned_fast_sweep):$(cat /sys/module/capstone_vm/parameters/supervisor_slots)
  ./namespace-check
  echo NAMESPACE_RC:$module:$?
  rmmod capstone_vm || exit 1
done
for module in good shared; do
  insmod $module.ko unpinned=1 unpinned_fast_sweep=1 supervisor_slots=32 || exit 1
  ./sweep-shared
  echo SHARED_RC:$module:$?
  rmmod capstone_vm || exit 1
done
rmmod namespace-probe || exit 1
insmod good.ko unpinned=1 unpinned_fast_sweep=1 supervisor_slots=32 || exit 1
for mode in growth stale; do
  CAPSTONE_VM_STATS=1 ./capstone-vexec node.dom $mode
  echo NODE_RC:$mode:$?
done
./capstone-vexec --stats
rmmod capstone_vm || exit 1
echo VIRTUAL_STAGED_DONE
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('qemu', 'images', 'module', 'mutant', 'shared-mutant', 'launcher', 'probes',
                 'application', 'work'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    stage = args.work / 'stage'
    stage.mkdir()
    inputs = {'good.ko': args.module, 'omit.ko': args.mutant, 'shared.ko': args.shared_mutant,
              'capstone-vexec': args.launcher, 'node.dom': args.application,
              **{n: args.probes / n for n in ('namespace-check', 'namespace-probe.ko', 'sweep-shared')}}
    for name, path in inputs.items():
        shutil.copy2(path, stage / name)
    (stage / 'gate.sh').write_text(SCRIPT)
    rc = subprocess.run(['python3', str(HERE.parent / 'run-staged.py'),
        '--qemu', str(args.qemu), '--images', str(args.images), '--stage', str(stage),
        '--work', str(args.work / 'guest'), '--disk-mib', '64', '--timeout', '600']).returncode
    log = (args.work / 'guest/serial.log').read_text(errors='replace').replace('\r', '')
    checks = {
        'complete': rc == 0 and 'VIRTUAL_STAGED_DONE' in log,
        'mode': log.count('SWEEP_MODE:Y:Y:32') == 2,
        'namespace': 'NAMESPACE_RC:good:0' in log,
        'namespace_omission_control': 'NAMESPACE_RC:omit:1' in log,
        'multiple_mappings': 'NAMESPACE mappings=4 pages=10 seeds=21 tag_after_clean=1 bytes_equal=1' in log,
        'scalar_clear': log.count('NAMESPACE direct_scalar_clear=1 bytes_equal=1') == 2,
        'shared_refusal': 'SHARED_RC:good:0' in log and 'SWEEP_SHARED refused=1 child_tag=1' in log,
        'shared_control': 'SHARED_RC:shared:1' in log and 'SWEEP_SHARED refused=0 child_tag=0' in log,
        'exclusive_retry': log.count('SWEEP_SHARED exclusive_clean=1') == 2,
        'growth': 'NODE_RC:growth:0' in log and 'NODE_GROWTH_OK' in log,
        'live_populations': log.count('NODE_LIVE:0:70000') == 2 and log.count('NODE_LIVE:1:70000') == 2,
        'stale': 'NODE_RC:stale:139' in log and 'domain fault cause=24' in log,
        'no_setup_refusal': 'thread create refused' not in log and 'Exec format error' not in log,
    }
    census = [json.loads(line) for line in log.splitlines() if line.startswith('{"version":1,')]
    checks['cleanup'] = len(census) == 1 and all(census[0][key] == 0 for key in
        ('live_domains', 'live_regions', 'live_bytes', 'nodes_live', 'nodes_retired'))
    checks['recycled'] = len(census) == 1 and census[0]['collections'] >= 4 and census[0]['nodes_reclaimed'] >= 280008
    inputs.update(qemu=args.qemu, **{n: args.images / n for n in ('Image', 'fw_jump.elf', 'rootfs.ext2')})
    record = {'checks': checks, 'sha256': {n: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for n, p in inputs.items()}}
    (args.work / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    for name, ok in checks.items():
        print(('PASS ' if ok else 'FAIL ') + name)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
