#!/usr/bin/env python3
"""tshark same-answers run: stock and protected (TSAPP_SUBLET=1) virtual images, one boot.

Every capture of the app port's oracle set runs `tshark -r C -V -n` (TZ=UTC, HOME and
WIRESHARK_CONFIG_DIR a fresh empty directory) on both images in one boot of the virtual platform
(runtime/virtual/run-staged.py), stock arm first. The guest reports per run: exit status, sha256
and length of stdout, sha256 and length of stderr less any TSAPP-HEAP line (host/run.py's rule),
a byte compare (cmp) against the native references staged beside it, and any launcher fault line.
The host computes the same references from native stock tshark (stdout) and the native minimal
build (stderr, TSAPP_MINIMAL), and judges:
  workload/pair captures: candidate stdout == native stock, stderr == native minimal, exit 0
  ntp: candidate stdout DIFFERS from native stock (outside the whitelist), stderr as above
  flips: native stock stdout on the flipped copy differs from the original (input sensitivity)
  protected == stock, stdout and stderr, per capture
An absent result is an ERROR, never a match.
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
import tempfile

CAPTURES = ['dhcp', 'dns_port', 'http', 'arp', 'dhcp.flip', 'dns_port.flip', 'http.flip', 'arp.flip',
            'ntp', 'dns-ooo']
ARMS = ['stock', 'protected']
KIT_FILES = {'qemu': 'qemu-system-riscv64', 'kernel': 'images/Image', 'firmware': 'images/fw_jump.elf',
             'rootfs': 'images/rootfs.ext2', 'launcher': 'adapter/capstone-vexec',
             'module': 'adapter/module/capstone_vm.ko'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fsha(path):
    return sha(Path(path).read_bytes())


def strip_heap(err):
    return b'\n'.join(l for l in err.split(b'\n') if not l.startswith(b'TSAPP-HEAP '))


def native(binary, capture):
    with tempfile.TemporaryDirectory() as cfg:
        env = dict(os.environ, TZ='UTC', HOME=cfg, WIRESHARK_CONFIG_DIR=cfg)
        r = subprocess.run([str(binary), '-r', str(capture), '-V', '-n'], capture_output=True, env=env)
    return r.returncode, r.stdout, r.stderr


GATE_HEAD = r'''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || { echo VIRTUAL_STAGED_DONE; exit 1; }
chmod 666 /dev/capstone-vm
export CAPSTONE_EXEC_DIAGNOSTICS=1
echo "@@TOOLS $(command -v sha256sum) $(command -v cmp) $(command -v wc)"
'''
GATE_RUN = r'''rm -rf /tmp/cfg; mkdir /tmp/cfg
echo "@@BEGIN {arm} {cap}"
HOME=/tmp/cfg WIRESHARK_CONFIG_DIR=/tmp/cfg TZ=UTC ./capstone-vexec ./{arm}.dom -r /mnt/vm/cap/{cap}.pcap -V -n > /tmp/out 2> /tmp/err.raw
rc=$?
grep -v '^TSAPP-HEAP ' /tmp/err.raw > /tmp/err
cmp -s /tmp/out /mnt/vm/ref/{cap}.stdout && so=SAME || so=DIFF
cmp -s /tmp/err /mnt/vm/ref/{cap}.stderr && se=SAME || se=DIFF
echo "@@RESULT {arm} {cap} rc=$rc out=$(sha256sum < /tmp/out | cut -c1-64) outlen=$(wc -c < /tmp/out) err=$(sha256sum < /tmp/err | cut -c1-64) errlen=$(wc -c < /tmp/err) cmp_out=$so cmp_err=$se"
grep -a 'domain fault cause=' /tmp/err.raw | sed 's/^/@@FAULT {arm} {cap} /'
if [ $so = DIFF ] && [ {cap} != ntp ]; then diff /mnt/vm/ref/{cap}.stdout /tmp/out | head -40 | sed 's/^/@@DIFF {arm} {cap} /'; fi
if [ $se = DIFF ]; then diff /mnt/vm/ref/{cap}.stderr /tmp/err | head -40 | sed 's/^/@@EDIFF {arm} {cap} /'; fi
echo "@@END {arm} {cap}"
'''


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--kit', type=Path, required=True, help='kit-sublet: qemu, images/, adapter/')
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--stock', type=Path, required=True, help='build-virtual.sh image dir, TSAPP_SUBLET unset')
    p.add_argument('--protected', type=Path, required=True, help='build-virtual.sh image dir, TSAPP_SUBLET=1')
    p.add_argument('--native-stock', type=Path, required=True)
    p.add_argument('--native-minimal', type=Path, required=True)
    p.add_argument('--captures', type=Path, required=True, help='a pristine test/captures directory')
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=5400)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    (stage / 'cap').mkdir(parents=True)
    (stage / 'ref').mkdir()
    record = dict(platform={k: fsha(a.kit / v) for k, v in KIT_FILES.items()}, images={}, native={}, captures={})
    for arm, d in (('stock', a.stock), ('protected', a.protected)):
        m = json.loads((d / 'manifest.json').read_text())
        image = d / 'tshark.dom'
        if fsha(image) != m['image_sha256']:
            sys.exit(f'{arm}: image does not match its manifest')
        record['images'][arm] = {k: m[k] for k in ('image_sha256', 'nested', 'profile', 'heap', 'recipe_revision',
                                                 'runtime_revision', 'runtime_dirty', 'compiler_sha256',
                                                 'runtime_archive_sha256', 'builtins_archive_sha256',
                                                 'stack_bytes', 'arena_bytes')}
        shutil.copy2(image, stage / f'{arm}.dom')
    if record['images']['stock']['nested'] != 'none' or record['images']['protected']['nested'] != 'wireshark':
        sys.exit('arms are not stock (nested=none) and protected (nested=wireshark)')
    for b in ('native_stock', 'native_minimal'):
        record['native'][b] = fsha(getattr(a, b))
    refs = {}
    for cap in CAPTURES:
        src = a.captures / (cap.removesuffix('.flip') + '.pcap')
        data = bytearray(src.read_bytes())
        if cap.endswith('.flip'):
            data[-8] ^= 1
        local = stage / 'cap' / (cap + '.pcap')
        local.write_bytes(data)
        rc, out, _ = native(a.native_stock, local)
        mrc, mout, merr = native(a.native_minimal, local)
        if rc or not out:
            sys.exit(f'{cap}: native stock did not complete (status {rc})')
        if mrc or not merr:
            sys.exit(f'{cap}: native minimal wrote no stderr (status {mrc}); an empty reference matches anything')
        (stage / 'ref' / (cap + '.stdout')).write_bytes(out)
        (stage / 'ref' / (cap + '.stderr')).write_bytes(merr)
        refs[cap] = dict(capture_sha256=sha(bytes(data)), native_stdout_sha256=sha(out), native_stdout_len=len(out),
                         minimal_stdout_sha256=sha(mout), minimal_stdout_equals_stock=mout == out,
                         minimal_stderr_sha256=sha(merr), minimal_stderr_len=len(merr))
        if cap.endswith('.flip'):
            orig = (stage / 'ref' / (cap.removesuffix('.flip') + '.stdout')).read_bytes()
            changed = sum(1 for l in subprocess.run(['diff', str(stage / 'ref' / (cap.removesuffix('.flip') + '.stdout')),
                                                     str(stage / 'ref' / (cap + '.stdout'))],
                                                    capture_output=True, text=True).stdout.splitlines()
                          if l[:1] in '<>')
            refs[cap]['flip_changes_native'] = orig != out
            refs[cap]['flip_diff_lines'] = changed
    record['captures'] = refs
    shutil.copy2(a.kit / KIT_FILES['launcher'], stage / 'capstone-vexec')
    (stage / 'capstone-vexec').chmod(0o755)
    shutil.copy2(a.kit / KIT_FILES['module'], stage / 'capstone_vm.ko')
    script = GATE_HEAD + ''.join(GATE_RUN.replace('{arm}', arm).replace('{cap}', cap)
                                 for arm in ARMS for cap in CAPTURES)
    script += 'rmmod capstone_vm\necho VIRTUAL_STAGED_DONE\n'
    (stage / 'gate.sh').write_text(script)
    record['gate_sha256'] = sha(script.encode())
    record['runner_sha256'] = fsha(__file__)
    record['run_staged_sha256'] = fsha(a.repo / 'capstone/runtime/virtual/run-staged.py')
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(a.kit) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    run = subprocess.run([sys.executable, str(a.repo / 'capstone/runtime/virtual/run-staged.py'),
                          '--qemu', str(a.kit / KIT_FILES['qemu']), '--images', str(a.kit / 'images'),
                          '--stage', str(stage), '--work', str(a.work / 'guest'), '--timeout', str(a.timeout)], env=env)
    log = (a.work / 'guest/serial.log').read_text(errors='replace').replace('\r', '')
    record['completed'] = run.returncode == 0
    record['tools'] = (re.findall(r'^@@TOOLS (.*)$', log, re.M) or ['absent'])[-1]
    results = {}
    for m in re.finditer(r'^@@RESULT (\S+) (\S+) rc=(\d+) out=(\S*) outlen=\s*(\d+) err=(\S*) errlen=\s*(\d+) '
                         r'cmp_out=(\S+) cmp_err=(\S+)$', log, re.M):
        arm, cap = m.group(1), m.group(2)
        results.setdefault(cap, {})[arm] = dict(rc=int(m.group(3)), stdout_sha256=m.group(4), stdout_len=int(m.group(5)),
                                                stderr_sha256=m.group(6), stderr_len=int(m.group(7)),
                                                cmp_stdout=m.group(8), cmp_stderr=m.group(9))
    for kind in ('FAULT', 'DIFF', 'EDIFF'):
        for m in re.finditer(rf'^@@{kind} (\S+) (\S+) (.*)$', log, re.M):
            r = results.setdefault(m.group(2), {}).setdefault(m.group(1), {})
            r.setdefault(kind.lower(), []).append(m.group(3))
    verdicts = {}
    ok = record['completed']
    for cap in CAPTURES:
        ref = refs[cap]
        v = {}
        for arm in ARMS:
            r = results.get(cap, {}).get(arm)
            if not r or 'rc' not in r:
                v[arm] = 'ERROR: no result'
                ok = False
                continue
            same_out = r['stdout_sha256'] == ref['native_stdout_sha256'] and r['cmp_stdout'] == 'SAME'
            same_err = r['stderr_sha256'] == ref['minimal_stderr_sha256'] and r['cmp_stderr'] == 'SAME'
            want_out = not same_out if cap == 'ntp' else same_out
            good = want_out and same_err and r['rc'] == 0 and not r.get('fault')
            if cap.endswith('.flip'):
                good &= ref['flip_changes_native']
            r['stdout_vs_native'] = 'MATCH' if same_out else 'DIFFERS'
            r['stderr_vs_minimal'] = 'MATCH' if same_err else 'DIFFERS'
            v[arm] = 'PASS' if good else 'FAIL'
            ok &= good
        s, q = results.get(cap, {}).get('stock', {}), results.get(cap, {}).get('protected', {})
        v['protected_equals_stock'] = bool(s and q and s.get('stdout_sha256') == q.get('stdout_sha256')
                                           and s.get('stderr_sha256') == q.get('stderr_sha256')
                                           and s.get('rc') == q.get('rc'))
        ok &= v['protected_equals_stock']
        verdicts[cap] = v
    record['results'] = results
    record['verdicts'] = verdicts
    record['status'] = 'PASS' if ok else 'FAIL'
    (a.work / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    for cap in CAPTURES:
        v = verdicts[cap]
        print(f"{cap:14s} stock {v['stock']:5s} protected {v['protected']:5s} protected==stock {v['protected_equals_stock']}")
    print('STATUS', record['status'])
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
