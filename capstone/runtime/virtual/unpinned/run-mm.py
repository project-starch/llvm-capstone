#!/usr/bin/env python3
"""Isolated one-hart swap-hook and fresh-namespace negative controls."""
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
swap=$(awk '$3 == "arch_prepare_to_swap" {print "0x"$1}' /proc/kallsyms)
test -n "$swap" || exit 1
insmod mm-probe.ko swap_prepare_address=$swap
echo MM_EXIT:$?
dmesg -c | grep RAW_MM
rmmod mm-probe || exit 1
insmod namespace-probe.ko || exit 1
for skip in 0 1; do
  insmod capstone_vm.ko unpinned=1 test_skip_namespace_scrub=$skip || exit 1
  echo "MM_MODE:$(cat /sys/module/capstone_vm/parameters/unpinned):$(cat /sys/module/capstone_vm/parameters/test_skip_namespace_scrub)"
  ./namespace-check
  echo NAMESPACE_EXIT:$skip:$?
  echo "NAMESPACE_SWEEP_PAGES:$(cat /sys/module/capstone_vm/parameters/unpinned_sweep_pages)"
  rmmod capstone_vm || exit 1
done
rmmod namespace-probe || exit 1
echo VIRTUAL_STAGED_DONE
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('qemu', 'images', 'module', 'probes', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    stage = a.work / 'stage'
    stage.mkdir(parents=True)
    paths = {name: a.probes / name for name in ('mm-probe.ko', 'namespace-probe.ko', 'namespace-check')}
    paths['capstone_vm.ko'] = a.module
    for name, src in paths.items():
        shutil.copy2(src, stage / name)
    (stage / 'gate.sh').write_text(SCRIPT)
    rc = subprocess.run(['python3', str(HERE.parent / 'run-staged.py'), '--qemu', str(a.qemu),
        '--images', str(a.images), '--stage', str(stage), '--work', str(a.work / 'guest'), '--timeout', '180']).returncode
    log = (a.work / 'guest/serial.log').read_text(errors='replace').replace('\r', '')
    checks = dict(completed=rc == 0, swap_hook='RAW_MM tagged_swap=-16 untagged_swap=0' in log,
        mm_probe='MM_EXIT:0' in log and 'RAW_MM PASS' in log,
        mode='MM_MODE:Y:N' in log and 'MM_MODE:Y:Y' in log,
        scalar_clear=log.count('NAMESPACE direct_scalar_clear=1 bytes_equal=1') == 2,
        scrub='NAMESPACE_EXIT:0:0' in log and 'NAMESPACE tag_before=1 tag_after=0 bytes_equal=1' in log,
        scrub_negative='NAMESPACE_EXIT:1:1' in log and 'NAMESPACE tag_before=1 tag_after=1 bytes_equal=1' in log)
    record = dict(checks=checks, sha256={name: hashlib.sha256(src.read_bytes()).hexdigest() for name, src in paths.items()})
    (a.work / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    for name, ok in checks.items():
        print(('PASS ' if ok else 'FAIL ') + name)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
