#!/usr/bin/env python3
"""Exercise internal NODES replies across stops and compare the launcher fallback."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

CONFIGS = {
    'base': '',
    'fallback': 'internal_nodes=0',
    'resched': 'test_inject=1 test_inject_at=5',
    'signal': 'test_inject=2 test_inject_at=5',
    'deadline': 'test_inject=3 test_inject_at=5',
    'retry': 'test_inject=2 test_inject_at=3 test_inject_every=7',
    'kill': 'test_inject=4 test_inject_at=5',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('qemu', 'images', 'adapter', 'module', 'application', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    inputs = {'qemu': a.qemu, 'launcher': a.adapter/'capstone-vexec',
              'module': a.module, 'application': a.application,
              **{n: a.images/n for n in ('Image', 'fw_jump.elf', 'rootfs.ext2')}}
    def hashes():
        return {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    before = hashes()
    for key, name in (('launcher', 'capstone-vexec'), ('module', 'capstone_vm.ko'),
                      ('application', 'node-services.dom')):
        shutil.copy2(inputs[key], stage/name)
    script = 'cd /mnt/vm || exit 1\nexport CAPSTONE_VM_STATS=1\necho 1 > /proc/sys/kernel/printk\n'
    for name, params in CONFIGS.items():
        script += f'''echo NODE_CASE:{name}
insmod capstone_vm.ko {params} || exit 1
./capstone-vexec node-services.dom
echo NODE_EXIT:$?
./capstone-vexec --stats
rmmod capstone_vm || exit 1
echo 1 > /proc/sys/vm/stat_refresh || exit 1
grep nr_foll_pin /proc/vmstat
dmesg -c | grep 'capstone_vm test unload'
'''
    script += 'echo VIRTUAL_STAGED_DONE\n'
    (stage/'gate.sh').write_text(script)
    run = subprocess.run([sys.executable, str(Path(__file__).with_name('run-staged.py')),
        '--qemu', str(a.qemu), '--images', str(a.images), '--stage', str(stage),
        '--work', str(a.work/'guest'), '--disk-mib', '64', '--timeout', '600'])
    log = (a.work/'guest/serial.log').read_text(errors='replace').replace('\r', '')
    checks = {'completed': run.returncode == 0, 'inputs_unchanged': before == hashes()}
    totals, observed = {}, {}
    for name in CONFIGS:
        marker = 'NODE_CASE:' + name + '\n'
        checks[name + '_case_once'] = log.count(marker) == 1
        # A missing case must not borrow another case's counters from the
        # whole log. Duplicate launches also invalidate an exactly-once check.
        part = log.split(marker)[1].split('NODE_CASE:')[0] if checks[name + '_case_once'] else ''
        counts = dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', part))
        pins = dict(re.findall(r'nr_foll_pin_(acquired|released) (\d+)', part))
        observed[name] = {'counters': counts, 'pins': pins}
        checks[name + '_pins'] = (int(pins.get('acquired', 0)) > 0 and
                                 pins.get('acquired') == pins.get('released'))
        census = [json.loads(l) for l in part.splitlines() if l.startswith('{"version":1,')]
        checks[name + '_cleanup'] = len(census) == 1 and all(
            census[0][k] == 0 for k in ('live_domains', 'live_regions', 'live_bytes'))
        if name == 'kill':
            checks[name + '_exit'] = part.count('NODE_EXIT:137\n') == 1 and 'NODE_SERVICES_OK' not in part
        else:
            checks[name + '_exit'] = (part.count('NODE_EXIT:0\n') == 1 and
                                     part.count('NODE_SERVICES_OK requests=128\n') == 1)
            internal = counts.get('internal_nodes', 0)
            external = counts.get('service12_nodes', 0)
            totals[name] = internal + external
            # A real quantum can expire before handling a request and leave
            # that request to the launcher, even with internal_nodes enabled.
            checks[name + '_route'] = (external >= 128 and internal == 0 if name == 'fallback'
                                       else internal > 0 and internal + external >= 128)
        if name not in ('base', 'fallback'):
            checks[name + '_injected'] = counts.get('injections', 0) > 0
            checks[name + '_stopped'] = counts.get('stops', 0) == counts.get('injections')
            checks[name + '_no_entry_pending'] = counts.get('violations') == 0
        checks[name + '_irqs_off'] = counts.get('checks', 0) > 0 and counts.get('checks_irqs_on') == 0
    checks['exactly_once'] = len(set(totals.values())) == 1 and len(totals) == len(CONFIGS) - 1
    result = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
              'node_totals': totals, 'observed': observed, 'sha256': before,
              'gate_sha256': hashlib.sha256(script.encode()).hexdigest(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.work/'result.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, ok in checks.items():
        print(('PASS ' if ok else 'FAIL ') + name)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
