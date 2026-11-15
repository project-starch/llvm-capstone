#!/usr/bin/env python3
"""tshark reuse distance at wmem, one instrumented arm, one boot (pre-registration 16d2171a3edc).

Each capture of the port's set runs `tshark -r C -V -n` (TZ=UTC, empty HOME/WIRESHARK_CONFIG_DIR)
once, as its own process. The image's exit-time `REUSE-GAP <boundary> ...` lines (one per wmem
boundary the process used) are taken from stderr; stdout's sha256 is kept as a check that the
instrumented runs print what the same-answers runs printed.

--set FILE replaces the port's ten (six captures of test/captures and four one-byte-flipped
copies) with the captures FILE lists, one path per line relative to --captures, each staged
unchanged under its own name (docs/plans/2026-11-15-wireshark-established-workload.md: the
release's test/captures). --rep N only labels result.json. Every process's exit status, stdout
hash and REUSE-GAP lines are recorded; "complete" needs exit 0, a REUSE-GAP line and no fault for
every capture. The markers' sed uses `|` as its delimiter, since a capture in a subdirectory
(mysql/, zabbix/) has a `/` in its name. With --set it also writes WORK/rep.json in the form
experiments/study/reuse-hooks/cpython/collect-reuse.py reads: each boundary's counters and bins
b1..b32 summed over the processes (a reuse distance never crosses a process), "rc" the number of
captures whose process did not exit 0, "error" the sum of the processes' observer errors.
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
./capstone-vexec --platform > /dev/null || { echo VIRTUAL_STAGED_DONE; exit 1; }
export CAPSTONE_EXEC_DIAGNOSTICS=1
'''
RUN = r'''rm -rf /tmp/cfg; mkdir /tmp/cfg
t0=$(date +%s)
HOME=/tmp/cfg WIRESHARK_CONFIG_DIR=/tmp/cfg TZ=UTC ./capstone-vexec ./tshark.dom -r /mnt/vm/cap/{file} -V -n > /tmp/out 2> /tmp/err.raw
rc=$?
echo "@@RUN {cap} rc=$rc out=$(sha256sum < /tmp/out | cut -c1-64)"
echo "@@TIME {cap} $(( $(date +%s) - t0 ))"
grep -a '^REUSE-GAP ' /tmp/err.raw | sed 's|^|@@RG {cap} |'
grep -a 'domain fault cause=' /tmp/err.raw | sed 's|^|@@FAULT {cap} |'
'''


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rep_record(a, manifest, record, runs, captures, complete, stage):
    """The collect-reuse.py form of one repetition: per boundary, sums over the processes."""
    summed = {}
    for c in captures:
        for name, b in runs.get(c, {}).get('boundaries', {}).items():
            s = summed.setdefault(name, {k: 0 for k in ('attempts', 'issues', 'releases', 'reuses',
                                                        'distinct_starts', 'error')}
                                  | {'bins': [0] * 32, 'capacity': b['capacity'], 'processes': 0})
            for k in ('attempts', 'issues', 'releases', 'reuses', 'distinct_starts', 'error'):
                s[k] += b[k]
            s['bins'] = [x + b[f'b{i + 1}'] for i, x in enumerate(s['bins'])]
            s['processes'] += 1
    kit = a.kit
    return {
        'program': 'wireshark', 'arm': {'none': 'stock', 'wireshark': 'protected'}[manifest['nested']],
        'rep': a.rep, 'completed': bool(complete), 'rc': sum(runs.get(c, {}).get('rc') != 0 for c in captures),
        'guest_seconds': sum(runs.get(c, {}).get('seconds') or 0 for c in captures),
        'reuse_gap': summed,
        'stdout_sha256': {c: runs.get(c, {}).get('stdout_sha256') for c in captures},
        'staged_sha256': {'tshark.dom': record['image_sha256'], 'capstone-vexec': fsha(stage / 'capstone-vexec'),
                          'gate.sh': record['gate_sha256'], 'set': record['set_sha256']},
        'platform_sha256': {n: fsha(kit / 'images' / n) for n in ('Image', 'fw_jump.elf', 'initramfs.cpio.gz')}
                           | {'qemu-system-riscv64': fsha(kit / 'qemu-system-riscv64')},
        'run_staged_sha256': fsha(a.repo / 'capstone/runtime/virtual/run-staged.py'),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--kit', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=5400)
    p.add_argument('--set', type=Path, help='captures to run instead of the port\'s ten, one per line')
    p.add_argument('--rep', type=int, default=1)
    a = p.parse_args()
    if a.set:
        captures = [c for c in a.set.read_text().split() if c]
        files = {c: c for c in captures}
        if len(set(captures)) != len(captures) or not all(re.fullmatch(r'[A-Za-z0-9_.+/-]+', c) for c in captures):
            sys.exit('capture names must be unique and shell-safe')
    else:
        captures = CAPTURES
        files = {c: c + '.pcap' for c in captures}
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    (stage / 'cap').mkdir(parents=True)
    m = json.loads((a.image / 'manifest.json').read_text())
    if fsha(a.image / 'tshark.dom') != m['image_sha256']:
        sys.exit('image does not match its manifest')
    shutil.copy2(a.image / 'tshark.dom', stage / 'tshark.dom')
    for cap in captures:
        if a.set:
            data = (a.captures / cap).read_bytes()
        else:
            data = bytearray((a.captures / (cap.removesuffix('.flip') + '.pcap')).read_bytes())
            if cap.endswith('.flip'):
                data[-8] ^= 1
        (stage / 'cap' / files[cap]).parent.mkdir(parents=True, exist_ok=True)
        (stage / 'cap' / files[cap]).write_bytes(data)
    shutil.copy2(a.kit / 'adapter/capstone-vexec', stage / 'capstone-vexec')
    (stage / 'capstone-vexec').chmod(0o755)
    script = HEAD + ''.join(RUN.replace('{file}', files[c]).replace('{cap}', c) for c in captures) + 'echo VIRTUAL_STAGED_DONE\n'
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
    for m2 in re.finditer(r'^@@TIME (\S+) (\d+)$', log, re.M):
        runs.setdefault(m2.group(1), {})['seconds'] = int(m2.group(2))
    for m2 in re.finditer(r'^@@FAULT (\S+) (.*)$', log, re.M):
        runs.setdefault(m2.group(1), {}).setdefault('fault', []).append(m2.group(2))
    record = dict(image_sha256=m['image_sha256'], nested=m['nested'], completed=run.returncode == 0, runs=runs,
                  gate_sha256=hashlib.sha256(script.encode()).hexdigest(), runner_sha256=fsha(__file__))
    if a.set:
        record.update(rep=a.rep, set_sha256=fsha(a.set),
                      captures_sha256={c: hashlib.sha256((stage / 'cap' / files[c]).read_bytes()).hexdigest()
                                       for c in captures})
    (a.work / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    complete = record['completed'] and all(runs.get(c, {}).get('rc') == 0 and runs[c].get('boundaries')
                                           and not runs[c].get('fault') for c in captures)
    if a.set:
        (a.work / 'rep.json').write_text(json.dumps(rep_record(a, m, record, runs, captures, complete, stage),
                                                    indent=1, sort_keys=True) + '\n')
    print('COMPLETE' if complete else 'INCOMPLETE')
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
