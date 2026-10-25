#!/usr/bin/env python3
"""tshark memory behaviour, one arm, one repetition, one boot (2026-10-11 pre-registration).

For each capture of the port's set, in one boot of the virtual platform:
  - `capstone-vexec --stats` immediately before and after the run (the module's global counters);
  - the run itself, `tshark -r C -V -n` (TZ=UTC, empty HOME/WIRESHARK_CONFIG_DIR) with
    CAPSTONE_VM_STATS=1, so the launcher prints its per-process CAPSTONE_VM_STATS line at exit;
  - peak resident memory from busybox `/usr/bin/time -f '%M'` around the launcher: the kernel's
    ru_maxrss for the waited child, which covers the launcher and every descendant it reaped (a
    /proc/<pid>/status VmHWM sample of the launcher's pid would see only its small spawner);
The guest also reports the stdout sha256 and the stderr sha256 with CAPSTONE_VM_STATS and
TSAPP-HEAP lines removed, as a check that the measured runs are the same-answers runs.
wmem exposes no allocator statistics: recorded as "not exposed".
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

CAPTURES = ['dhcp', 'dns_port', 'http', 'arp', 'dhcp.flip', 'dns_port.flip', 'http.flip', 'arp.flip',
            'ntp', 'dns-ooo']

HEAD = r'''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || { echo VIRTUAL_STAGED_DONE; exit 1; }
chmod 666 /dev/capstone-vm
export CAPSTONE_EXEC_DIAGNOSTICS=1
echo "@@STATS boot $(./capstone-vexec --stats)"
'''
RUN = r'''rm -rf /tmp/cfg; mkdir /tmp/cfg
echo "@@STATS before {cap} $(./capstone-vexec --stats)"
HOME=/tmp/cfg WIRESHARK_CONFIG_DIR=/tmp/cfg TZ=UTC CAPSTONE_VM_STATS=1 /usr/bin/time -f 'TIME-MAXRSS %M ELAPSED %e' ./capstone-vexec ./tshark.dom -r /mnt/vm/cap/{cap}.pcap -V -n > /tmp/out 2> /tmp/err.raw
rc=$?
echo "@@STATS after {cap} $(./capstone-vexec --stats)"
grep -v -e '^CAPSTONE_VM_STATS ' -e '^TSAPP-HEAP ' -e '^TIME-MAXRSS ' /tmp/err.raw > /tmp/err
echo "@@RUN {cap} rc=$rc $(grep '^TIME-MAXRSS ' /tmp/err.raw | tail -1) out=$(sha256sum < /tmp/out | cut -c1-64) err=$(sha256sum < /tmp/err | cut -c1-64)"
grep -a '^CAPSTONE_VM_STATS ' /tmp/err.raw | sed 's/^/@@VMSTATS {cap} /'
grep -a 'domain fault cause=' /tmp/err.raw | sed 's/^/@@FAULT {cap} /'
'''


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--kit', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True, help='build-virtual.sh image dir')
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=5400)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    (stage / 'cap').mkdir(parents=True)
    m = json.loads((a.image / 'manifest.json').read_text())
    if fsha(a.image / 'tshark.dom') != m['image_sha256']:
        sys.exit('image does not match its manifest')
    shutil.copy2(a.image / 'tshark.dom', stage / 'tshark.dom')
    for cap in CAPTURES:
        data = bytearray((a.captures / (cap.removesuffix('.flip') + '.pcap')).read_bytes())
        if cap.endswith('.flip'):
            data[-8] ^= 1
        (stage / 'cap' / (cap + '.pcap')).write_bytes(data)
    shutil.copy2(a.kit / 'adapter/capstone-vexec', stage / 'capstone-vexec')
    (stage / 'capstone-vexec').chmod(0o755)
    shutil.copy2(a.kit / 'adapter/module/capstone_vm.ko', stage / 'capstone_vm.ko')
    script = HEAD + ''.join(RUN.replace('{cap}', c) for c in CAPTURES) + 'rmmod capstone_vm\necho VIRTUAL_STAGED_DONE\n'
    (stage / 'gate.sh').write_text(script)
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(a.kit) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    run = subprocess.run([sys.executable, str(a.repo / 'capstone/runtime/virtual/run-staged.py'),
                          '--qemu', str(a.kit / 'qemu-system-riscv64'), '--images', str(a.kit / 'images'),
                          '--stage', str(stage), '--work', str(a.work / 'guest'), '--timeout', str(a.timeout)], env=env)
    log = (a.work / 'guest/serial.log').read_text(errors='replace').replace('\r', '')
    stats = {}
    for m2 in re.finditer(r'^@@STATS (boot|before|after) ?(\S*) (\{.*\})$', log, re.M):
        stats.setdefault(m2.group(2) or 'boot', {})[m2.group(1)] = json.loads(m2.group(3))
    runs = {}
    for m2 in re.finditer(r'^@@RUN (\S+) rc=(\d+) (?:TIME-MAXRSS (\d+) ELAPSED (\S+) )?out=(\S+) err=(\S+)$', log, re.M):
        runs[m2.group(1)] = dict(rc=int(m2.group(2)),
                                 max_rss_kb=int(m2.group(3)) if m2.group(3) else 'not measured',
                                 elapsed_s=float(m2.group(4)) if m2.group(4) else None,
                                 stdout_sha256=m2.group(5), stderr_sha256=m2.group(6))
    for m2 in re.finditer(r'^@@VMSTATS (\S+) CAPSTONE_VM_STATS (.*)$', log, re.M):
        runs.setdefault(m2.group(1), {})['vm_stats'] = {k: int(v) for k, v in
                                                        (x.split('=') for x in m2.group(2).split())}
    for m2 in re.finditer(r'^@@FAULT (\S+) (.*)$', log, re.M):
        runs.setdefault(m2.group(1), {}).setdefault('fault', []).append(m2.group(2))
    for cap in CAPTURES:
        s = stats.get(cap, {})
        if 'before' in s and 'after' in s:
            runs.setdefault(cap, {})['stats_delta'] = {k: s['after'][k] - s['before'][k] for k in s['after']
                                                       if isinstance(s['after'][k], int)}
            runs[cap]['stats_before'] = s['before']
            runs[cap]['stats_after'] = s['after']
    record = dict(image_sha256=m['image_sha256'], nested=m['nested'], completed=run.returncode == 0,
                  boot_stats=stats.get('boot', {}).get('boot'), runs=runs, gate_sha256=hashlib.sha256(script.encode()).hexdigest(),
                  runner_sha256=fsha(__file__), wmem_statistics='not exposed')
    (a.work / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    complete = record['completed'] and all(c in runs and 'rc' in runs[c] and 'stats_delta' in runs[c] for c in CAPTURES)
    print('COMPLETE' if complete else 'INCOMPLETE')
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
