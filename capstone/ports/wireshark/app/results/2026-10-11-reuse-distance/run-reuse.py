#!/usr/bin/env python3
"""tshark reuse distance at wmem, one instrumented arm, one boot (pre-registration 16d2171a3edc).

Each capture of the port's set runs `tshark -r C -V -n` (TZ=UTC, empty HOME/WIRESHARK_CONFIG_DIR)
once, as its own process. The image's exit-time `REUSE-GAP <boundary> ...` lines (one per wmem
boundary the process used) are taken from stderr; stdout's sha256 is kept as a check that the
instrumented runs print what the same-answers runs printed.
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
'''
RUN = r'''rm -rf /tmp/cfg; mkdir /tmp/cfg
HOME=/tmp/cfg WIRESHARK_CONFIG_DIR=/tmp/cfg TZ=UTC ./capstone-vexec ./tshark.dom -r /mnt/vm/cap/{cap}.pcap -V -n > /tmp/out 2> /tmp/err.raw
rc=$?
echo "@@RUN {cap} rc=$rc out=$(sha256sum < /tmp/out | cut -c1-64)"
grep -a '^REUSE-GAP ' /tmp/err.raw | sed 's/^/@@RG {cap} /'
grep -a 'domain fault cause=' /tmp/err.raw | sed 's/^/@@FAULT {cap} /'
'''


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--kit', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
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
    runs = {}
    for m2 in re.finditer(r'^@@RUN (\S+) rc=(\d+) out=(\S+)$', log, re.M):
        runs[m2.group(1)] = dict(rc=int(m2.group(2)), stdout_sha256=m2.group(3), boundaries={})
    for m2 in re.finditer(r'^@@RG (\S+) REUSE-GAP (\S+) (.*)$', log, re.M):
        kv = dict(x.split('=', 1) for x in m2.group(3).split())
        b = {k: (kv[k] if k == 'arm' else int(kv[k])) for k in kv}
        runs.setdefault(m2.group(1), {}).setdefault('boundaries', {})[m2.group(2)] = b
    for m2 in re.finditer(r'^@@FAULT (\S+) (.*)$', log, re.M):
        runs.setdefault(m2.group(1), {}).setdefault('fault', []).append(m2.group(2))
    record = dict(image_sha256=m['image_sha256'], nested=m['nested'], completed=run.returncode == 0, runs=runs,
                  gate_sha256=hashlib.sha256(script.encode()).hexdigest(), runner_sha256=fsha(__file__))
    (a.work / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    complete = record['completed'] and all(runs.get(c, {}).get('rc') == 0 and runs[c].get('boundaries')
                                           and not runs[c].get('fault') for c in CAPTURES)
    print('COMPLETE' if complete else 'INCOMPLETE')
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
