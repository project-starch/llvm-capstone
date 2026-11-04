#!/usr/bin/env python3
"""Completeness of the fast sweep against the old scanner, with early-stop mutants.

Run through the shared gate queue. --stage supplies: plain modules good.ko,
early_pte.ko and early_vma.ko; CV_TEST_HOOKS modules hgood.ko, hearly_pte.ko and
hearly_vma.ko; namespace-probe.ko, namespace-check and sweep-span (the fixture
from the namespace probe, plus sweep-span.c here).

early_pte stops a VMA at the end of the first PTE table that holds no tagged page;
early_vma stops the whole sweep after the first VMA that yielded a page. Both are
applied only to the fast scanner. Three kinds of evidence:

 1. sweep-span and namespace-check, which seed tags in later PTE tables and later
    VMAs, must pass on good.ko and sweep-span must FAIL on both early-stop mutants
    (tags survive the scrub);
 2. the test build's differential (unpinned_sweep_diff=1) runs the other scanner
    on the same mm at every sweep and counts mismatching held-page lists; it must
    read runs > 0, pages > 0 and zero mismatches on hgood.ko in both directions
    (fast active with old as reference, and old active with fast as reference)
    and a nonzero mismatch count on both early-stop mutants in both directions;
 3. the same differential, on a real program, is the perlbench row of the fix
    round (see the lane report); this runner is the deterministic part.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
PARAMS = 'runs active other mismatch digest errors'.split()
SCRIPT = '''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod namespace-probe.ko || exit 1
p=/sys/module/capstone_vm/parameters
for module in good early_pte early_vma; do
  insmod $module.ko unpinned=1 unpinned_fast_sweep=1 supervisor_slots=32 || exit 1
  echo SPAN_BEGIN:$module
  ./sweep-span
  echo SPAN_RC:$module:$?
  ./namespace-check
  echo NAMESPACE_RC:$module:$?
  rmmod capstone_vm || exit 1
done
for spec in hgood:1 hearly_pte:1 hearly_vma:1 hgood:0 hearly_pte:0 hearly_vma:0; do
  module=${spec%:*}; fast=${spec#*:}
  insmod $module.ko unpinned=1 unpinned_fast_sweep=$fast unpinned_sweep_diff=1 supervisor_slots=32 || exit 1
  echo DIFF_BEGIN:$module:$fast
  ./sweep-span
  echo SPAN_RC:$module:$fast:$?
  ./namespace-check
  echo NAMESPACE_RC:$module:$fast:$?
  echo DIFF:$module:$fast:$(cat $p/diff_runs):$(cat $p/diff_active_pages):$(cat $p/diff_other_pages):$(cat $p/diff_mismatch):$(cat $p/diff_digest_mismatch):$(cat $p/diff_other_errors)
  rmmod capstone_vm || exit 1
done
rmmod namespace-probe || exit 1
echo VIRTUAL_STAGED_DONE
'''
NAMES = ('good.ko', 'early_pte.ko', 'early_vma.ko', 'hgood.ko', 'hearly_pte.ko', 'hearly_vma.ko',
         'namespace-probe.ko', 'namespace-check', 'sweep-span')


def diff_row(log, module, fast):
    m = re.search(r'^DIFF:%s:%d:(\d+):(\d+):(\d+):(\d+):(\d+):(\d+)$' % (module, fast), log, re.M)
    return dict(zip(PARAMS, map(int, m.groups()))) if m else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('qemu', 'images', 'stage', 'work'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    stage = args.work / 'stage'
    stage.mkdir()
    inputs = {name: args.stage / name for name in NAMES}
    for name, path in inputs.items():
        shutil.copy2(path, stage / name)
    (stage / 'gate.sh').write_text(SCRIPT)
    rc = subprocess.run(['python3', str(HERE.parent / 'run-staged.py'),
        '--qemu', str(args.qemu), '--images', str(args.images), '--stage', str(stage),
        '--work', str(args.work / 'guest'), '--disk-mib', '64', '--timeout', '600']).returncode
    log = (args.work / 'guest/serial.log').read_text(errors='replace').replace('\r', '')
    checks = {'complete': rc == 0 and 'VIRTUAL_STAGED_DONE' in log,
              'no_setup_refusal': 'thread create refused' not in log and 'Exec format error' not in log}
    for module, span_rc in (('good', 0), ('early_pte', 1), ('early_vma', 1)):
        checks['span_' + module] = f'SPAN_RC:{module}:{span_rc}' in log
    checks['span_good_line'] = 'SPAN tables=4 vmas_with_tags=3 seeds=6 tag_after_clean=1 bytes_equal=1' in log
    checks['namespace_good'] = 'NAMESPACE_RC:good:0' in log
    checks['span_mutants_leave_tags'] = log.count('SPAN_TAGGED ') >= 2
    for fast in (1, 0):
        row = diff_row(log, 'hgood', fast)
        checks[f'diff_good_fast{fast}'] = bool(row and row['runs'] >= 4 and row['active'] > 0 and
            row['active'] == row['other'] and row['mismatch'] == 0 and row['digest'] == 0 and row['errors'] == 0)
        for module in ('hearly_pte', 'hearly_vma'):
            row = diff_row(log, module, fast)
            checks[f'diff_{module}_fast{fast}'] = bool(row and row['runs'] >= 4 and row['mismatch'] > 0 and
                                                       row['active'] != row['other'])
    for fast in (1, 0):
        for module in ('hgood', 'hearly_pte', 'hearly_vma'):
            row = diff_row(log, module, fast)
            checks[f'sweep_reached_{module}_fast{fast}'] = bool(row and row['runs'] > 0)
    inputs.update(qemu=args.qemu, **{n: args.images / n for n in ('Image', 'fw_jump.elf', 'rootfs.ext2')})
    record = {'checks': checks, 'diff_rows': {f'{m}:{f}': diff_row(log, m, f)
                                              for m in ('hgood', 'hearly_pte', 'hearly_vma') for f in (1, 0)},
              'sha256': {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in inputs.items()}}
    (args.work / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    for name, ok in checks.items():
        print(('PASS ' if ok else 'FAIL ') + name)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
