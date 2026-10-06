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
for mode in stale bounds retired; do
  ./capstone-vexec contract.dom "$mode" > /tmp/fault.out 2>&1
  echo VM_EXIT:$mode:$?
  cat /tmp/fault.out
done
./capstone-vexec contract.dom spin > /tmp/spin.out 2>&1 & spin=$!
sleep 2
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
rmmod capstone_vm
echo VM_EXIT:cleanup:$?
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
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--omit-application', action='store_true')
    a = p.parse_args()
    work = Path(tempfile.mkdtemp(prefix='virtual-app.', dir=os.environ['CAPSTONE_TMP_ROOT']))
    stage = work/'stage'; stage.mkdir()
    inputs = {'qemu': a.qemu, 'launcher': a.adapter/'capstone-vexec',
              'module': a.adapter/'module/capstone_vm.ko', 'application': a.application,
              'sqlite': a.sqlite,
              'mruby': a.mruby,
              **{n: a.images/n for n in ('Image', 'fw_jump.elf', 'rootfs.ext2')}}
    shutil.copyfile(inputs['launcher'], stage/'capstone-vexec')
    (stage/'capstone-vexec').chmod(0o755)
    shutil.copyfile(inputs['module'], stage/'capstone_vm.ko')
    if not a.omit_application: shutil.copyfile(a.application, stage/'contract.dom')
    shutil.copyfile(a.sqlite, stage/'sqlite3.dom')
    shutil.copyfile(a.mruby, stage/'mruby.dom')
    (stage/'gate.sh').write_text(SCRIPT)
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
        deadline = time.monotonic() + 300
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
    for name, code in [('normal', 0), ('threads', 0), ('stale', 139), ('bounds', 139), ('retired', 139),
                       ('preempt', 143), ('parallel_one', 0), ('parallel_two', 0),
                       ('sqlite_write', 0), ('sqlite_read', 0), ('mruby', 0), ('cleanup', 0)]:
        tests[name] = lines.count(f'VM_EXIT:{name}:{code}') == 1
    tests['normal_output'] = sum(s.startswith('VIRTUAL_APPLICATION_OK ') for s in lines) == 3
    tests['threads_output'] = (lines.count('VIRTUAL_THREAD_CHILD') == 1 and
                               lines.count('VIRTUAL_THREADS_OK shared_mm lifetime_root quantum') == 1)
    causes = [int(m[1]) for s in lines if (m := re.match(r'capstone-exec: domain fault cause=(\d+) ', s))]
    # Loading a saved revoked pointer clears its tag; the later dereference
    # is therefore cause 24. The direct live-register node check has its own
    # cause-25 instruction gate.
    tests['fault_causes'] = causes == [24, 28, 24]
    tests['same_va_reuse'] = lines.count('VIRTUAL_VA_REUSED') == 1
    tests['spin_entered'] = lines.count('VIRTUAL_SPIN_READY') == 1
    stats = [dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', s))
             for s in lines if s.startswith('CAPSTONE_VM_STATS ')]
    tests['linux_pages_released'] = len(stats) == 6 and all(s['peak'] > s['pages'] + 300 for s in stats[:3])
    tests['linux_demand_faults'] = len(stats) == 6 and all(s['faults'] >= 384 for s in stats[:3])
    tests['sqlite_results'] = lines.count('SQLITE_SUM=5050') == 1 and lines.count('SQLITE_PERSIST=100') == 1
    tests['mruby_results'] = lines.count('MRUBY_SUM=5050') == 1 and lines.count('MRUBY_FILE=virtual') == 1
    tests['completed'] = lines.count('VIRTUAL_RUNTIME_DONE') == 1
    result = {'status': 'PASS' if all(tests.values()) else 'FAIL', 'tests': tests,
              'scope': 'one hart; private anonymous arenas; same-mm virtual threads',
              'control_omit_application': a.omit_application,
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
