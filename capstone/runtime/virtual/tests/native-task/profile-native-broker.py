#!/usr/bin/env python3
"""Separate route census and flat kernel-PC profile for native SQLite speedtest1.

Run under run-quick-gate.py. Inputs are external; no benchmark source is fetched.
"""
import argparse
import bisect
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
for key in ['qemu', 'images', 'firmware', 'plugin', 'binary', 'profrun', 'reference', 'out']:
    p.add_argument('--' + key, type=Path, required=True)
p.add_argument('--locks', type=int, choices=[0, 1], default=0)
p.add_argument('--fast', type=int, choices=[0, 1], default=1)
p.add_argument('--stat', type=int, choices=[0, 1], default=0)
p.add_argument('--seek', type=int, choices=[0, 1], default=0)
p.add_argument('--fd', type=int, choices=[0, 1], default=0)
p.add_argument('--close', type=int, choices=[0, 1], default=0)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
virtual = Path(__file__).resolve().parents[2]
inputs = {k: hashlib.sha256(getattr(a, k).read_bytes()).hexdigest()
          for k in ['qemu', 'firmware', 'plugin', 'binary', 'profrun', 'reference']}
inputs.update({k: hashlib.sha256((a.images/k).read_bytes()).hexdigest()
               for k in ['Image', 'System.map', 'rootfs.ext2']})
(a.out/'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
reference = [x for x in a.reference.read_text().splitlines() if x.startswith('STUDY-ORACLE ')]
assert len(reference) == 32
flags = ('capstone_vm.native_task=1 capstone_vm.native_io=1 capstone_vm.native_fast=1 '
         'capstone_vm.dom_binfmt=1 capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1 '
         'capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1 capstone_vm.supervisor_slots=32')
flags += f' capstone_vm.native_locks={a.locks} capstone_vm.native_stat={a.stat} capstone_vm.native_seek={a.seek} capstone_vm.native_fd={a.fd} capstone_vm.native_close={a.close} capstone_vm.native_fast={a.fast}'
results = {}
for mode in ['routes', 'profile']:
    d = a.out/mode; stage = d/'stage'; stage.mkdir(parents=True)
    (d/'profile').mkdir()
    shutil.copy2(a.binary, stage/'sqlite.dom'); shutil.copy2(a.profrun, stage/'profrun')
    (stage/'gate.sh').write_text('''#!/bin/sh
set -eu
trap 'echo BROKER_CENSUS_EXIT:$?; echo VIRTUAL_STAGED_DONE' EXIT
cp -a /mnt/vm /tmp/broker-census
cd /tmp/broker-census
[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status
echo 0 > /proc/sys/kernel/randomize_va_space
killall syslogd klogd 2>/dev/null || true
chmod +x profrun sqlite.dom
./profrun ./sqlite.dom --size 1 st.db >sqlite.out 2>sqlite.err
cat sqlite.out sqlite.err
echo SQLITE_OUTPUT_END
dmesg
''')
    cmd = ['python3', str(virtual/'run-staged.py'), '--qemu', str(a.qemu),
           '--images', str(a.images), '--firmware', str(a.firmware), '--stage', str(stage),
           '--work', str(d/'vm'), '--disk-mib', '128', '--memory', '4G', '--timeout', '240',
           '--icount', '0', '--kernel-arg', flags + f' capstone_vm.native_route_trace={int(mode == "routes")}']
    if mode == 'profile': cmd += ['--plugin', str(a.plugin), '--plugin-arg', 'out='+str(d/'profile')]
    (d/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA='x-capstone-supervisor-slots=32,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true')
    with (d/'runner.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (d/'vm/serial.log').read_text(errors='replace').replace('\r', '')
    assert 'BROKER_CENSUS_EXIT:0\n' in log and 'VIRTUAL_STAGED_DONE' in log
    assert not re.search(r'WARNING:|BUG:|Oops|Kernel panic', log)
    assert [x for x in log.splitlines() if x.startswith('STUDY-ORACLE ')] == reference
    if mode == 'routes':
        calls = re.findall(r'native_route route=(\w+) nr=(\d+) count=(\d+) result=(-?\d+)', log.split('SQLITE_OUTPUT_END\n', 1)[0])
        assert calls
        routes = Counter((route, int(nr)) for route,nr,_,_ in calls)
        errors = Counter((int(nr),int(rc)) for _,nr,_,rc in calls if int(rc) < 0)
        results['routes'] = [dict(route=route, nr=nr, calls=n) for (route,nr),n in routes.most_common()]
        results['errors'] = [dict(nr=nr, result=rc, calls=n) for (nr,rc),n in errors.most_common()]
    else:
        windows = list((d/'profile').glob('window-*.tsv'))
        assert len(windows) == 1 and not list((d/'profile').glob('*.invalid'))
        symbols = {}; by_name = {}
        for line in (a.images/'System.map').read_text().splitlines():
            fields = line.split()
            if len(fields) < 3: continue
            address, kind, name = fields[:3]; address = int(address, 16); by_name[name] = address
            if kind in ['t','T']: symbols[address] = name
        addresses = sorted(symbols); counts = Counter(); kernel = other = 0
        for line in windows[0].read_text().splitlines():
            if line.startswith('#'): continue
            fields = line.split(); pc, attempts = int(fields[0],16), int(fields[3])
            if by_name['_stext'] <= pc < by_name['_etext']:
                kernel += attempts
                index = bisect.bisect_right(addresses, pc)-1
                counts[symbols[addresses[index]] if index >= 0 else '<unknown>'] += attempts
            else: other += attempts
        results['profile'] = dict(kernel_attempts=kernel, other_attempts=other,
            total_attempts=kernel+other, symbols=[dict(symbol=name, attempts=n,
            kernel_percent=100*n/kernel) for name,n in counts.most_common()])
    print('PASS', mode, flush=True)
result = dict(status='PASS', inputs=inputs, configuration=dict(native_locks=a.locks, native_stat=a.stat, native_seek=a.seek, native_fd=a.fd, native_close=a.close, native_fast=a.fast), oracle_phases_per_run=32, **results,
    limitations=['One separate diagnostic route run and one untraced instruction-profile run.',
                 'Flat PC attribution, not call stacks or inclusive cost per syscall family; inlining may merge costs.',
                 'Route frequency alone does not establish which family consumes the most kernel work.',
                 'Attempted QEMU guest instructions, not hardware timing; no runtime changes or speedup claim.'])
(a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(routes=results['routes'], top_kernel_symbols=results['profile']['symbols'][:15])), flush=True)
