#!/usr/bin/env python3
"""Isolated one-hart swap-hook and fresh-namespace negative controls."""
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
swap=$(awk '$3 == "arch_prepare_to_swap" {print "0x"$1}' /proc/kallsyms)
test -n "$swap" || exit 1
insmod mm-probe.ko swap_prepare_address=$swap
echo MM_EXIT:$?
dmesg -c | grep RAW_MM
rmmod mm-probe || exit 1
insmod namespace-probe.ko || exit 1
# arm NAME  fixture  module parameters. good: the unbroken module; skip: no scrub at all; pages1 /
# granules1: a scrub that stops after its first page / first granule of each page. The multi-seed
# fixture must fail every broken arm; the single-seed fixture (the first version) passes the partial
# ones, which is why it is kept only as that record.
arm() {
  name=$1; fixture=$2; shift 2
  insmod capstone_vm.ko unpinned=1 "$@" || exit 1
  echo "MM_ARM:$name:$fixture:$(cat /sys/module/capstone_vm/parameters/unpinned)"
  ./namespace-check $fixture
  echo "NAMESPACE_EXIT:$name:$fixture:$?"
  echo "NAMESPACE_SWEEP_PAGES:$name:$fixture:$(cat /sys/module/capstone_vm/parameters/unpinned_sweep_pages)"
  rmmod capstone_vm || exit 1
}
arm good multi
arm skip multi test_skip_namespace_scrub=1
arm pages1 multi test_skip_scrub_after_pages=1
arm granules1 multi test_skip_scrub_after_granules=1
arm good single
arm pages1 single test_skip_scrub_after_pages=1
arm granules1 single test_skip_scrub_after_granules=1
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
    arms = {}
    for block in re.finditer(r'^MM_ARM:(\w+):(\w+):(\w)\n(.*?)^NAMESPACE_SWEEP_PAGES:\1:\2:(\d+)$', log, re.M | re.S):
        name, fixture, mode, text, swept = block.groups()
        m = re.search(r'NAMESPACE mappings=(\d+) pages=(\d+) seeds=(\d+) tag_after_clean=(\d) bytes_equal=(\d)', text)
        arms[(name, fixture)] = dict(
            mode=mode, swept=int(swept), exit=int(re.search(r'^NAMESPACE_EXIT:\w+:\w+:(\d+)$', text + log[block.end():block.end() + 80], re.M)[1]),
            seen=m is not None, mappings=int(m[1]) if m else 0, pages=int(m[2]) if m else 0, seeds=int(m[3]) if m else 0,
            clean=bool(m and m[4] == '1'), equal=bool(m and m[5] == '1'),
            direct='NAMESPACE direct_scalar_clear=1 bytes_equal=1' in text)

    def arm(name, fixture):
        return arms.get((name, fixture), {})

    good, single = arm('good', 'multi'), arm('good', 'single')
    broken = [arm(n, 'multi') for n in ('skip', 'pages1', 'granules1')]
    checks = dict(completed=rc == 0, swap_hook='RAW_MM tagged_swap=-16 untagged_swap=0' in log,
        mm_probe='MM_EXIT:0' in log and 'RAW_MM PASS' in log,
        mode=len(arms) == 7 and all(r['mode'] == 'Y' for r in arms.values()),
        # eleven pages of four mappings, thirty-one granules: the fixture reached what it claims
        fixture_reach=good.get('mappings') == 4 and good.get('pages') == 11 and good.get('seeds') == 31,
        scrub_all_pages=bool(good.get('seen') and good['exit'] == 0 and good['clean'] and good['equal']),
        # the module's own count of the tagged pages it swept is every page the fixture seeded
        sweep_count=good.get('swept') == good.get('pages') and good.get('pages', 0) > 1,
        scalar_clear=all(r.get('direct') for r in arms.values()) and len(arms) == 7,
        # each broken variant leaves seeded tags behind and fails the same oracle (exit 1), not an
        # incidental error before it
        broken_variants_fail=all(r.get('seen') and r['exit'] == 1 and not r['clean'] for r in broken),
        # (the sweep counter counts the tagged pages the module HELD, so a partial scrub still
        # reports them all: only the skipped scrub reports none)
        skipped_scrub_held_nothing=arm('skip', 'multi').get('swept') == 0,
        # the first fixture could not see a partial scrub
        single_seed_blind=bool(single.get('exit') == 0 and arm('pages1', 'single').get('exit') == 0
                               and arm('granules1', 'single').get('exit') == 0))
    record = dict(checks=checks, arms={f'{n}:{f}': r for (n, f), r in arms.items()}, sha256={name: hashlib.sha256(src.read_bytes()).hexdigest() for name, src in paths.items()})
    (a.work / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    for name, ok in checks.items():
        print(('PASS ' if ok else 'FAIL ') + name)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
