#!/usr/bin/env python3
"""Growing node tables, bounded quota failures, reuse, and stale denial."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
SCRIPT = r'''#!/bin/sh
cd /mnt/vm || exit 1
export CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1
insmod capstone_vm.ko node_initial_pages=4 node_batch_pages=16 || exit 1
for mode in growth stale; do
  ./capstone-vexec node.dom "$mode" > /tmp/node.out 2>&1
  echo NODE_EXIT:$mode:$?
  cat /tmp/node.out
done
./capstone-vexec --stats
rmmod capstone_vm || exit 1
insmod capstone_vm.ko node_initial_pages=4 node_batch_pages=16 node_max_pages=6 || exit 1
./capstone-vexec node.dom budget > /tmp/node.out 2>&1
echo NODE_EXIT:budget:$?
cat /tmp/node.out
./capstone-vexec node.dom growth > /tmp/node.out 2>&1
echo NODE_EXIT:quota:$?
cat /tmp/node.out
./capstone-vexec --stats
rmmod capstone_vm || exit 1
echo VIRTUAL_STAGED_DONE
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qemu', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--application', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'; stage.mkdir()
    inputs = {'qemu': a.qemu, 'application': a.application,
              'launcher': a.adapter/'capstone-vexec',
              'module': a.adapter/'module/capstone_vm.ko',
              **{n: a.images/n for n in ('Image', 'fw_jump.elf', 'rootfs.ext2')}}
    hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    for src, name in ((inputs['application'], 'node.dom'),
                      (inputs['launcher'], 'capstone-vexec'),
                      (inputs['module'], 'capstone_vm.ko')):
        shutil.copy2(src, stage/name)
    (stage/'gate.sh').write_text(SCRIPT)
    run = subprocess.run([os.environ.get('PYTHON', 'python3'),
        str(HERE/'run-staged.py'), '--qemu', str(a.qemu), '--images', str(a.images),
        '--stage', str(stage), '--work', str(a.work/'guest'), '--disk-mib', '64',
        '--timeout', str(a.timeout)])
    log = (a.work/'guest/serial.log').read_text(errors='replace')
    stats = [dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
             for line in log.splitlines() if line.startswith('CAPSTONE_VM_STATS ')]
    census = [json.loads(line) for line in log.splitlines() if line.startswith('{"version":1,')]
    checks = {'completed': run.returncode == 0,
              'growth': 'NODE_EXIT:growth:0' in log and 'NODE_GROWTH_OK' in log,
              'two_live_populations': log.count('NODE_LIVE:0:70000') == 2 and log.count('NODE_LIVE:1:70000') == 2,
              'stale': 'NODE_EXIT:stale:139' in log and 'NODE_STALE_ACCESS' in log,
              'budget_errno': 'NODE_EXIT:budget:0' in log and 'NODE_BUDGET_OK' in log,
              'quota_fault': 'NODE_EXIT:quota:139' in log,
              'causes': re.findall(r'^capstone-exec: domain fault cause=(\d+) ', log, re.M) == ['24', '30'],
              # --stats opens its own empty context: only its initial table
              # should remain visible after each tested process has exited.
              'cleanup': len(census) == 2 and all(
                  s['live_domains'] == s['live_regions'] == s['live_bytes'] ==
                  s['nodes_live'] == s['nodes_retired'] == 0 and
                  s['node_capacity'] == 1024 and s['node_bytes'] == 7 * 4096
                  for s in census),
              'no_contract_failure': 'NODE_FAIL:' not in log}
    if stats:
        s = stats[0]
        checks.update(grew_past_old_limit=s['node_capacity'] > 65536 and s['node_growths'] > 0,
                      recycled=s['collections'] >= 2 and s['reclaimed'] >= 140000,
                      bounded_reuse=s['node_capacity'] < 100000,
                      actual_metadata_bytes=s['node_capacity'] * 16 < s['node_bytes'] < s['node_capacity'] * 17)
    else:
        checks['statistics'] = False
    checks['inputs_unchanged'] = hashes == {
        k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    result = {'schema': 1, 'status': 'PASS' if all(checks.values()) else 'FAIL',
              'checks': checks, 'stats': stats,
              'sha256': hashes,
              'source_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), HERE/'node-contract.c',
                                          HERE/'run-staged.py']},
              'gate_sha256': hashlib.sha256(SCRIPT.encode()).hexdigest(),
              'scope': 'one hart, trusted Linux, process-local paged node store'}
    (a.work/'result.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, ok in checks.items(): print(('PASS ' if ok else 'FAIL ') + name)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
