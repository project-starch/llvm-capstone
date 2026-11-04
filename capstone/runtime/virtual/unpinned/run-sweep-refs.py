#!/usr/bin/env python3
"""Real-page reference and PROT_NONE checks, including a missing-put mutant.

Run through the shared gate queue. --stage supplies good.ko, leak.ko, omit.ko,
sweep-ref-probe.ko, sweep-refs and namespace-probe.ko.
Only the binaries named below are copied; the probe and client are test-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
SCRIPT = '''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod namespace-probe.ko || exit 1
insmod sweep-ref-probe.ko || exit 1
for module in good leak omit; do
  insmod $module.ko unpinned=1 unpinned_fast_sweep=1 supervisor_slots=32 || exit 1
  echo REFS_BEGIN:$module
  ./sweep-refs
  echo REFS_END:$module:$?
  rmmod capstone_vm || exit 1
done
rmmod sweep_ref_probe || exit 1
rmmod namespace_probe || exit 1
echo VIRTUAL_STAGED_DONE
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('qemu', 'images', 'stage', 'work'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    stage = args.work/'stage'; stage.mkdir()
    inputs = {name: args.stage/name for name in
              ('good.ko', 'leak.ko', 'omit.ko', 'namespace-probe.ko', 'sweep-ref-probe.ko', 'sweep-refs')}
    for name, path in inputs.items():
        shutil.copy2(path, stage/name)
    (stage/'gate.sh').write_text(SCRIPT)
    rc = subprocess.run(['python3', str(HERE.parent/'run-staged.py'),
        '--qemu', str(args.qemu), '--images', str(args.images), '--stage', str(stage),
        '--work', str(args.work/'guest'), '--disk-mib', '64', '--timeout', '180']).returncode
    log = (args.work/'guest/serial.log').read_text(errors='replace').replace('\r', '')
    checks = {'complete': rc == 0 and 'VIRTUAL_STAGED_DONE' in log,
              'no_setup_refusal': 'thread create refused' not in log and 'Exec format error' not in log}
    for module, expected_exit, leak, clean, refused in (
            ('good', 0, 0, 1, 1), ('leak', 1, 1, 1, 1), ('omit', 1, 0, 0, 0)):
        part = re.search(r'^REFS_BEGIN:'+module+r'\n(.*?)^REFS_END:'+module+r':(\d+)\n', log, re.M|re.S)
        checks[module+'_exit'] = bool(part and int(part[2]) == expected_exit)
        success = re.search(r'SWEEP_REFS success_before=(\d+) success_after=(\d+) protnone_clean=(\d+) same_page=(\d+)', part[1] if part else '')
        failure = re.search(r'SWEEP_REFS failure_before=(\d+) failure_after=(\d+) refused=(\d+) tag_preserved=(\d+) same_page=(\d+)', part[1] if part else '')
        shared = re.search(r'SWEEP_REFS shared_before=(\d+) shared_after=(\d+) same_page=(\d+) mappings=(\d+)', part[1] if part else '')
        checks[module+'_protnone'] = bool(success and int(success[3]) == clean)
        checks[module+'_success_refs'] = bool(success and int(success[1]) == 2 and int(success[2]) == 2+leak and int(success[4]) == 1)
        checks[module+'_failure_refs'] = bool(failure and int(failure[1]) == 2 and int(failure[2]) == 2+leak and int(failure[5]) == 1)
        checks[module+'_shared_refs'] = bool(shared and tuple(map(int, shared.groups())) == (3, 3, 1, 2))
        checks[module+'_rollback'] = bool(failure and int(failure[3]) == refused and int(failure[4]) == 1)
    inputs.update(qemu=args.qemu, **{n: args.images/n for n in ('Image','fw_jump.elf','rootfs.ext2')})
    record = {'checks': checks, 'sha256': {n: hashlib.sha256(p.read_bytes()).hexdigest() for n,p in inputs.items()}}
    (args.work/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    for name, ok in checks.items(): print(('PASS ' if ok else 'FAIL ')+name)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
