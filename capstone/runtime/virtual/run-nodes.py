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

import memory_mode
import native_platform

HERE = Path(__file__).resolve().parent
PARAMETERS = '/sys/module/capstone_vm/parameters'
SCRIPT = r'''#!/bin/sh
cd /mnt/vm || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1
export CAPSTONE_NODE_COUNT=__NODE_COUNT__
__ADMIT__
__MODE_ECHO__
echo "NODE_PAGES:initial=$(cat __PARAMETERS__/node_initial_pages) batch=$(cat __PARAMETERS__/node_batch_pages) max=$(cat __PARAMETERS__/node_max_pages)"
__RUNS__
__LEAVE__
echo VIRTUAL_STAGED_DONE
'''
GROWTH_RUNS = r'''for mode in growth stale; do
  ./capstone-vexec node.dom "$mode" > /tmp/node.out 2>&1
  echo NODE_EXIT:$mode:$?
  cat /tmp/node.out
done'''
QUOTA_RUNS = r'''./capstone-vexec node.dom budget > /tmp/node.out 2>&1
echo NODE_EXIT:budget:$?
cat /tmp/node.out
./capstone-vexec node.dom growth > /tmp/node.out 2>&1
echo NODE_EXIT:quota:$?
cat /tmp/node.out'''
# The node table's page quotas are kernel parameters, read-only after boot. Each set of quotas is
# therefore one boot: its suffix for the stage and work directories, its runs, its kernel words
# and the quotas the guest must then report.
BOOTS = (('', GROWTH_RUNS, ('node_initial_pages=4', 'node_batch_pages=16'), 'initial=4 batch=16 max=0'),
         ('-quota', QUOTA_RUNS, ('node_initial_pages=4', 'node_batch_pages=16', 'node_max_pages=6'),
          'initial=4 batch=16 max=6'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qemu', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--application', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=600)
    # vp2 caps the flat lifetime table at 65536 records (a 16-bit node index), so 70000 cannot be
    # the count everywhere. Default unchanged for QEMU; silicon and its QEMU prediction run 60000.
    # The value is substituted into the gate script AND recorded, and the live-population check
    # below is built from it -- a hardcoded 70000 there would silently pass or fail on a different
    # count rather than comparing like with like.
    p.add_argument('--unpinned', action='store_true', help='require the adapter to report unpinned memory (the native adapter has no other mode)')
    p.add_argument('--node-count', type=int, default=70000)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    inputs = {'qemu': a.qemu, 'application': a.application,
              'launcher': a.adapter/native_platform.LAUNCHER,
              **{n: a.images/n for n in ('Image', 'fw_jump.elf', 'rootfs.ext2')}}
    hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    gate_text, log, completed = '', '', True
    for suffix, runs, words, quotas in BOOTS:
        stage = a.work / ('stage' + suffix); stage.mkdir()
        for src, name in ((inputs['application'], 'node.dom'),
                          (inputs['launcher'], native_platform.LAUNCHER)):
            shutil.copy2(src, stage/name)
        script = (SCRIPT.replace('__NODE_COUNT__', str(a.node_count))
                  .replace('__ADMIT__', native_platform.admit())
                  .replace('__MODE_ECHO__', memory_mode.echo('NODE_'))
                  .replace('__PARAMETERS__', PARAMETERS)
                  .replace('__RUNS__', runs)
                  .replace('__LEAVE__', native_platform.leave('NODE_')))
        assert '__NODE_COUNT__' not in script, 'node count not substituted'
        (stage/'gate.sh').write_text(script)
        gate_text += script
        guest = a.work / ('guest' + suffix)
        run = subprocess.run([os.environ.get('PYTHON', 'python3'),
            str(HERE/'run-staged.py'), '--qemu', str(a.qemu), '--images', str(a.images),
            '--stage', str(stage), '--work', str(guest), '--disk-mib', '64',
            '--timeout', str(a.timeout)] + [x for word in words for x in ('--kernel-arg', 'capstone_vm.' + word)])
        completed = completed and run.returncode == 0
        serial = guest/'serial.log'
        log += serial.read_text(errors='replace').replace('\r', '') if serial.exists() else ''
    stats = [dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
             for line in log.splitlines() if line.startswith('CAPSTONE_VM_STATS ')]
    census = [json.loads(line) for line in log.splitlines() if line.startswith('{"version":1,')]
    nm = Path(os.environ['CAPSTONE_LLVM_BIN']) / 'llvm-nm'
    symbols = {}
    for line in subprocess.check_output([str(nm), '--defined-only', str(a.application)], text=True).splitlines():
        fields = line.split()
        if len(fields) == 3:
            symbols[fields[2]] = int(fields[0], 16)
    headers = subprocess.check_output([str(nm.with_name('llvm-readelf')), '-l', str(a.application)], text=True)
    loads = [int(line.split()[2], 16) for line in headers.splitlines() if line.strip().startswith('LOAD ')]
    # the launcher prints ` ra= fp=` after the address when the platform wrote the register
    # slots (2026-10-27); the site is the pc either way, as an offset from the code range's
    # base, where the kernel mapped the image's lowest load address
    stale_sites = re.findall(
        r'^capstone-exec: domain fault cause=24 pc=(0x[0-9a-f]+) '
        r'address=0x[0-9a-f]+(?: ra=0x[0-9a-f]+ fp=0x[0-9a-f]+)? entry=0x[0-9a-f]+ code=(0x[0-9a-f]+)-', log, re.M)
    checks = {'completed': completed,
              'node_quotas': re.findall(r'^NODE_PAGES:(.*)$', log, re.M) == [quotas for *_, quotas in BOOTS],
              'growth': 'NODE_EXIT:growth:0' in log and 'NODE_GROWTH_OK' in log,
              'two_live_populations': (log.count('NODE_LIVE:0:%d' % a.node_count) == 2 and
                                       log.count('NODE_LIVE:1:%d' % a.node_count) == 2),
              # the domain prints the count it used; if that line is missing the run is not
              # comparable with any other, so its absence is a failure and not a zero
              'node_count_reported': log.count('NODE_COUNT_USED %d' % a.node_count) >= 1,
              'stale': 'NODE_EXIT:stale:139' in log and 'NODE_STALE_ACCESS' in log,
              'stale_site': len(stale_sites) == 1 and
                  'cap_node_stale' in symbols and bool(loads) and
                  int(stale_sites[0][0], 16) - int(stale_sites[0][1], 16) ==
                  symbols['cap_node_stale'] - min(loads),
              'budget_errno': 'NODE_EXIT:budget:0' in log and 'NODE_BUDGET_OK' in log,
              'quota_fault': 'NODE_EXIT:quota:139' in log,
              'causes': re.findall(r'^capstone-exec: domain fault cause=(\d+) ', log, re.M) == ['24', '30'],
              'no_loadable_module': log.count('NODE_CLEANUP:0\n') == len(BOOTS),
              'memory_mode_verified': memory_mode.verified(log, a.unpinned, len(BOOTS)),
              'no_contract_failure': 'NODE_FAIL:' not in log}
    # The remaining checks read the context's statistics: the launcher's CAPSTONE_VM_STATS line at
    # an image's exit, and the census a statistics context printed after the runs. A native task
    # has no launcher loop and the kernel publishes neither, so the record names these
    # observations as not available.
    not_observable = []
    if census:
        # The statistics context is empty: only its initial table
        # should remain visible after each tested process has exited.
        checks['cleanup'] = len(census) == 2 and all(
            s['live_domains'] == s['live_regions'] == s['live_bytes'] ==
            s['nodes_live'] == s['nodes_retired'] == 0 and
            s['node_capacity'] == 1024 and s['node_bytes'] == 7 * 4096
            for s in census)
    else:
        not_observable += ['cleanup']
    if stats:
        s = stats[0]
        checks.update(grew_past_old_limit=s['node_capacity'] > 65536 and s['node_growths'] > 0,
                      recycled=s['collections'] >= 2 and s['reclaimed'] >= 140000,
                      bounded_reuse=s['node_capacity'] < 100000,
                      actual_metadata_bytes=s['node_capacity'] * 16 < s['node_bytes'] < s['node_capacity'] * 17)
    else:
        not_observable += ['grew_past_old_limit', 'recycled', 'bounded_reuse', 'actual_metadata_bytes']
    checks['inputs_unchanged'] = hashes == {
        k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    result = {'schema': 1, 'status': 'PASS' if all(checks.values()) else 'FAIL',
              'checks': checks, 'not_observable': not_observable, 'stats': stats,
              'sha256': hashes,
              'source_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), HERE/'node-contract.c',
                                          HERE/'run-staged.py', HERE/'native_platform.py']},
              'gate_sha256': hashlib.sha256(gate_text.encode()).hexdigest(),
              'node_count': a.node_count,
              'unpinned': a.unpinned,
              'scope': 'one hart, trusted Linux, process-local paged node store'}
    (a.work/'result.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, ok in checks.items(): print(('PASS ' if ok else 'FAIL ') + name)
    for name in not_observable: print('NOT OBSERVABLE ' + name)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
