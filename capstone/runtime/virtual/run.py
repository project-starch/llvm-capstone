#!/usr/bin/env python3
"""Qualify the virtual application adapter on the existing Linux image."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

SCRIPT = r'''#!/bin/sh
cd /mnt/vm || exit 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_VM_CONTRACT=environment CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1
./capstone-vexec contract.dom ok argument > /tmp/ok.out 2>&1
echo VM_EXIT:normal:$?
cat /tmp/ok.out
./capstone-vexec contract.dom threads > /tmp/threads.out 2>&1
echo VM_EXIT:threads:$?
cat /tmp/threads.out
for mode in vm linear linear-stale heap-threads sparse-churn stale bounds retired vm-ro-write vm-none-read vm-guard vm-padding sparse-stale; do
  ./capstone-vexec contract.dom "$mode" 2>&1
  echo VM_EXIT:$mode:$?
done
./capstone-vexec contract.dom spin > /tmp/spin.out 2>&1 & spin=$!
# QEMU startup includes Linux image loading and the first delegated rounds;
# allow the child to publish its readiness before testing signal termination.
sleep 5
kill -TERM "$spin" 2>/dev/null
wait "$spin"; echo VM_EXIT:preempt:$?
cat /tmp/spin.out
./capstone-vexec contract.dom ok argument > /tmp/one.out 2>&1 & one=$!
./capstone-vexec contract.dom ok argument > /tmp/two.out 2>&1 & two=$!
wait "$one"; echo VM_EXIT:parallel_one:$?
wait "$two"; echo VM_EXIT:parallel_two:$?
cat /tmp/one.out /tmp/two.out
./capstone-vexec sqlite3.dom /tmp/virtual.db "CREATE TABLE t(n); WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x WHERE n<100) INSERT INTO t SELECT n FROM x; SELECT 'SQLITE_SUM='||sum(n) FROM t;" > /tmp/sqlite.out 2>&1
echo VM_EXIT:sqlite_write:$?
cat /tmp/sqlite.out
./capstone-vexec sqlite3.dom /tmp/virtual.db "SELECT 'SQLITE_PERSIST='||count(*) FROM t;" > /tmp/sqlite.out 2>&1
echo VM_EXIT:sqlite_read:$?
cat /tmp/sqlite.out
./capstone-vexec mruby.dom -e 'puts "MRUBY_SUM=#{(1..100).inject(0){|s,n| s+n}}"; File.open("/tmp/virtual-ruby", "w"){|f| f.write("virtual")}; puts "MRUBY_FILE=#{File.read("/tmp/virtual-ruby")}"' > /tmp/mruby.out 2>&1
echo VM_EXIT:mruby:$?
cat /tmp/mruby.out
if [ -f perl.dom ]; then
  ./capstone-vexec perl.dom -e 'my $sum = 0; $sum += $_ for 1..100; open my $w, ">", "/tmp/virtual-perl" or die $!; print $w "virtual"; close $w; open my $r, "<", "/tmp/virtual-perl" or die $!; my $text = <$r>; close $r; print "PERL_VIRTUAL_OK sum=$sum file=$text\n"' > /tmp/perl.out 2>&1
  echo VM_EXIT:perl:$?
  cat /tmp/perl.out
  if [ -f perl-smoke.pl ]; then
    mkdir -p /tmp/virtual-perl-files
    cp /mnt/vm/perl-smoke.pl /tmp/virtual-perl-files/smoke.pl
    CAPSTONE_PERL_SMOKE_OBJECTS=2000 ./capstone-vexec perl.dom /mnt/vm/perl-smoke.pl /tmp/virtual-perl-files > /tmp/perl-smoke.out 2>&1
    echo VM_EXIT:perl_smoke:$?
    cat /tmp/perl-smoke.out
  fi
fi
rmmod capstone_vm
echo VM_EXIT:cleanup:$?
if [ -f legacy.dom ]; then
  ./capstone-vexec legacy.dom ok argument > /tmp/legacy.out 2>&1
  echo VM_EXIT:legacy:$?
  cat /tmp/legacy.out
fi
echo VIRTUAL_RUNTIME_DONE
'''

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qemu', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--application', type=Path, required=True)
    p.add_argument('--sqlite', type=Path, required=True)
    p.add_argument('--mruby', type=Path, required=True)
    p.add_argument('--perl', type=Path)
    p.add_argument('--perl-smoke', type=Path)
    p.add_argument('--legacy-application', type=Path)
    p.add_argument('--skip-recycling', action='store_true', help='Omit the two long recycling cases; recorded explicitly')
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--omit-application', action='store_true')
    a = p.parse_args()
    work = Path(tempfile.mkdtemp(prefix='virtual-app.', dir=os.environ['CAPSTONE_TMP_ROOT']))
    stage = work/'stage'; stage.mkdir()
    inputs = {'qemu': a.qemu, 'launcher': a.adapter/'capstone-vexec',
              'module': a.adapter/'module/capstone_vm.ko', 'application': a.application,
              'sqlite': a.sqlite,
              'mruby': a.mruby,
              **{n: a.images/n for n in ('Image', 'fw_jump.elf', 'rootfs.ext2')}}
    if a.legacy_application: inputs['legacy'] = a.legacy_application
    if a.perl: inputs['perl'] = a.perl
    if a.perl_smoke: inputs['perl_smoke'] = a.perl_smoke
    shutil.copyfile(inputs['launcher'], stage/'capstone-vexec')
    (stage/'capstone-vexec').chmod(0o755)
    shutil.copyfile(inputs['module'], stage/'capstone_vm.ko')
    if not a.omit_application: shutil.copyfile(a.application, stage/'contract.dom')
    shutil.copyfile(a.sqlite, stage/'sqlite3.dom')
    shutil.copyfile(a.mruby, stage/'mruby.dom')
    if a.perl: shutil.copyfile(a.perl, stage/'perl.dom')
    if a.perl_smoke: shutil.copyfile(a.perl_smoke, stage/'perl-smoke.pl')
    if a.legacy_application: shutil.copyfile(a.legacy_application, stage/'legacy.dom')
    script = SCRIPT
    if a.skip_recycling:
        script = script.replace(' sparse-churn', '').replace(' sparse-stale', '')
    (stage/'gate.sh').write_text(script)
    disk = work/'stage.ext4'
    with disk.open('wb') as f: f.truncate(64 << 20)
    subprocess.run(['/sbin/mkfs.ext4', '-F', '-q', '-d', str(stage), str(disk)], check=True)
    cmd = [str(a.qemu), '-M', 'virt-capstone', '-m', '2G', '-smp', '1',
           '-nographic', '-monitor', 'none', '-serial', 'stdio', '-bios', str(inputs['fw_jump.elf']),
           '-kernel', str(inputs['Image']), '-append', 'root=/dev/vda ro', '-snapshot',
           '-drive', f'file={inputs["rootfs.ext2"]},format=raw,id=hd0',
           '-device', 'virtio-blk-device,drive=hd0',
           '-drive', f'file={disk},format=raw,id=gate,readonly=on',
           '-device', 'virtio-blk-device,drive=gate', '-cpu',
           'rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true']
    with open(os.environ['CAPSTONE_QEMU_LOCK'], 'a+b') as lock, (work/'serial.log').open('wb') as log:
        fcntl.flock(lock, fcntl.LOCK_EX)
        guest = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=0)
        out = b''; login = sent = False
        deadline = time.monotonic() + a.timeout
        try:
            while time.monotonic() < deadline:
                if not select.select([guest.stdout], [], [], 1)[0]:
                    if guest.poll() is not None: break
                    continue
                data = os.read(guest.stdout.fileno(), 65536)
                if not data: break
                log.write(data); log.flush(); out += data
                if not login and b'login:' in out:
                    guest.stdin.write(b'root\n'); guest.stdin.flush(); login = True; out = b''
                if login and not sent and b'# ' in out:
                    guest.stdin.write(b'mkdir -p /mnt/vm; mount -t ext4 -o ro /dev/vdb /mnt/vm && sh /mnt/vm/gate.sh\n')
                    guest.stdin.flush(); sent = True; out = b''
                if b'\nVIRTUAL_RUNTIME_DONE\r\n' in out: break
                if b'Kernel panic' in out or b'Oops' in out: break
        finally:
            guest.terminate()
            try: guest.wait(timeout=5)
            except subprocess.TimeoutExpired: guest.kill(); guest.wait()
    lines = [s.strip() for s in out.decode(errors='replace').splitlines()]
    tests = {}
    for name, code in [('normal', 0), ('threads', 0), ('vm', 0), ('linear', 0), ('linear-stale', 139), ('heap-threads', 0), ('vm-ro-write', 139), ('vm-none-read', 139), ('vm-guard', 139), ('vm-padding', 139), ('sparse-churn', 0), ('sparse-stale', 139), ('stale', 139), ('bounds', 139), ('retired', 139),
                       ('preempt', 143), ('parallel_one', 0), ('parallel_two', 0),
                       ('sqlite_write', 0), ('sqlite_read', 0), ('mruby', 0), ('cleanup', 0)]:
        if a.skip_recycling and name in ('sparse-churn', 'sparse-stale'): continue
        tests[name] = lines.count(f'VM_EXIT:{name}:{code}') == 1
    if a.legacy_application:
        tests['legacy_abi_rejected'] = lines.count('VM_EXIT:legacy:126') == 1 and any('Exec format error' in s for s in lines)
    if a.perl:
        tests['perl'] = lines.count('VM_EXIT:perl:0') == 1
        tests['perl_output'] = lines.count('PERL_VIRTUAL_OK sum=5050 file=virtual') == 1
    if a.perl_smoke:
        tests['perl_smoke'] = lines.count('VM_EXIT:perl_smoke:0') == 1
        smoke_markers = (
            'P1 hello', 'P2 array 2000 4002000', 'P3 hash 3000 2999',
            'P4 string 2390 ab0ab1ab2a', 'P5 churn ok', 'P6 fib 6765',
            'P7 deep 500', 'P8 eval caught msg',
            'P9 num 4.50 2.5 1.18059162071741e+21',
            'P10 sort apple banana fig pear',
            'P11 regex The quick | The slow brown fox | 2',
            'P12 closure 5', 'P13 ref ARRAY REF 2', 'P14 method Counter 3',
            'P15 file 50 last-ok', 'P16 dir 1', 'P17 pack 1,2,3 03.14',
            'SMOKE_DONE')
        tests['perl_smoke_output'] = all(lines.count(marker) == 1 for marker in smoke_markers)
    tests['normal_output'] = sum(s.startswith('VIRTUAL_APPLICATION_OK ') for s in lines) == 3
    tests['threads_output'] = (lines.count('VIRTUAL_THREAD_CHILD') == 1 and
                               lines.count('VIRTUAL_THREADS_OK shared_mm lifetime_root quantum') == 1)
    causes = [int(m[1]) for s in lines if (m := re.match(r'capstone-exec: domain fault cause=(\d+) ', s))]
    # Loading a saved revoked pointer clears its tag; the later dereference
    # is therefore cause 24. The direct live-register node check has its own
    # cause-25 instruction gate.
    tests['fault_causes'] = causes == [24, 24, 28, 24, 15, 13, 13, 28] + ([] if a.skip_recycling else [24])
    symbols = {}
    nm = Path(os.environ['CAPSTONE_LLVM_BIN']) / 'llvm-nm'
    for line in subprocess.check_output([str(nm), '--defined-only', str(a.application)], text=True).splitlines():
        fields = line.split()
        if len(fields) == 3:
            symbols[fields[2]] = int(fields[0], 16)
    fault_sites = [tuple(int(v, 16) for v in m.groups()) for s in lines
                   if (m := re.search(r'domain fault cause=\d+ pc=0x([0-9a-f]+).* entry=0x([0-9a-f]+)', s))]
    names = ('ro', 'none', 'guard', 'padding') + (() if a.skip_recycling else ('collected',))
    tests['linear_output'] = lines.count('VIRTUAL_LINEAR_OK') == 1
    tests['linear_denial_site'] = bool(fault_sites) and fault_sites[0][0] - fault_sites[0][1] == symbols.get('cap_vm_fault_linear', -1) - symbols['domain_main']
    tests['vm_denial_sites'] = len(fault_sites) == 4 + len(names) and all(
        pc - entry == symbols.get('cap_vm_fault_' + name, -1) - symbols['domain_main']
        for name, (pc, entry) in zip(names, fault_sites[4:]))
    tests['vm_contract'] = lines.count('VIRTUAL_VM_OK protections guards requested_length unused_pages execute rollback') == 1
    tests['heap_threads'] = lines.count('VIRTUAL_HEAP_THREADS_OK shared_ownership metadata_growth arena_growth') == 1
    tests['vm_denial_operations'] = all(lines.count('VIRTUAL_VM_ACCESS:' + op) == 1
                                      for op in ('write_ro', 'read_none', 'guard', 'padding') + (() if a.skip_recycling else ('collected',)))
    tests['same_va_reuse'] = lines.count('VIRTUAL_VA_REUSED') == 1
    tests['spin_entered'] = lines.count('VIRTUAL_SPIN_READY') == 1
    stats = [dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', s))
             for s in lines if s.startswith('CAPSTONE_VM_STATS ')]
    # normal, threads, VM policy, linear loan, shared heap, optional sparse churn,
    # two normal launches, SQLite x2, mruby. Preserve the original page oracle.
    start = 5 if a.skip_recycling else 6
    base_stats = stats[:2] + stats[start:start + 5]
    release_stats = [s for s in base_stats if s['peak'] > s['pages'] + 300]
    demand_stats = [s for s in base_stats if s.get('faults', 0) >= 384]
    expected_stats = (10 if a.skip_recycling else 11) + (1 if a.perl else 0) + (1 if a.perl_smoke else 0)
    tests['linux_pages_released'] = len(stats) == expected_stats and len(release_stats) == 3
    tests['linux_demand_faults'] = len(stats) == expected_stats and len(demand_stats) == 3
    tests['unused_pages_not_populated'] = len(stats) >= 4 and stats[2]['peak'] - stats[2]['pages'] < 32
    tests['shared_heap_pages_released'] = len(stats) >= 5 and stats[4]['peak'] - stats[4]['pages'] >= 768
    if not a.skip_recycling:
        sparse = stats[5] if len(stats) >= 6 else {}
        # The fixed table happened to collect ~195k IDs in three sweeps.
        # A growing table has different collection boundaries. Require reuse
        # to explain allocations beyond capacity, with more than two tablefuls
        # allocated and no more than the old 1-MiB metadata budget. Retired
        # nodes in the unfinished final collection epoch need not be reclaimed.
        capacity = sparse.get('node_capacity', sparse.get('node_bytes', 0) // 16)
        tests['sparse_recycling'] = (lines.count('VIRTUAL_SPARSE_CHURN_OK allocations=200000 protected_stale_tags') == 1
                                 and capacity > 0 and sparse['collections'] > 0
                                 and sparse['nodes'] > 2 * capacity
                                 and sparse['reclaimed'] >= sparse['nodes'] - capacity
                                 and sparse['node_bytes'] <= 1048576 and sparse['peak'] < 2048)
    tests['sqlite_results'] = lines.count('SQLITE_SUM=5050') == 1 and lines.count('SQLITE_PERSIST=100') == 1
    tests['mruby_results'] = lines.count('MRUBY_SUM=5050') == 1 and lines.count('MRUBY_FILE=virtual') == 1
    tests['completed'] = lines.count('VIRTUAL_RUNTIME_DONE') == 1
    result = {'status': 'PASS' if all(tests.values()) else 'FAIL', 'tests': tests,
              'scope': 'one hart; private anonymous arenas; same-mm virtual threads',
              'control_omit_application': a.omit_application,
              'recycling_cases_enabled': not a.skip_recycling,
              'sha256': {n: digest(path) for n, path in inputs.items()},
              'source_sha256': {str(p.relative_to(HERE)): digest(p) for p in sorted(HERE.rglob('*'))
                               if p.is_file() and p.suffix in ('.c', '.h', '.S', '.sh', '.py')},
              'stats': stats}
    a.record.parent.mkdir(parents=True, exist_ok=True)
    a.record.write_text(json.dumps(result, indent=2) + '\n')
    print(f'{result["status"]}: {work / "serial.log"}')
    for s in lines:
        if s.startswith(('VM_EXIT:', 'VIRTUAL_', 'CAPSTONE_VM_', 'capstone-vexec:', 'VM_CONTRACT_', 'SQLITE_', 'MRUBY_')): print(s)
    return 0 if result['status'] == 'PASS' else 1

if __name__ == '__main__': raise SystemExit(main())
