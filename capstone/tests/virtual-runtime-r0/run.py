#!/usr/bin/env python3
"""Boot unchanged Linux plus the R0 module; require exact guest verdicts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qemu', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--module', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--c-entry', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--omit-probe', action='store_true')
    a = p.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix='runtime-r0-linux.',
                         dir=os.environ.get('CAPSTONE_TMP_ROOT', '/tmp')))
    stage = scratch/'stage'; stage.mkdir()
    shutil.copyfile(a.module, stage/'runtime_r0.ko')
    shutil.copyfile(a.c_entry, stage/'c-entry.bin')
    if not a.omit_probe:
        shutil.copyfile(a.probe, stage/'probe'); (stage/'probe').chmod(0o755)
    disk = scratch/'probe.ext4'
    with disk.open('wb') as f: f.truncate(32*1024*1024)
    subprocess.run(['/sbin/mkfs.ext4', '-F', '-q', '-d', str(stage), str(disk)], check=True)
    command = [str(a.qemu), '-M', 'virt-capstone', '-m', '2G', '-smp', '1',
               '-nographic', '-monitor', 'none', '-serial', 'stdio',
               '-bios', str(a.images/'fw_jump.elf'), '-kernel', str(a.images/'Image'),
               '-append', 'root=/dev/vda ro', '-snapshot', '-drive',
               f'file={a.images}/rootfs.ext2,format=raw,id=hd0',
               '-device', 'virtio-blk-device,drive=hd0', '-drive',
               f'file={disk},format=raw,id=probe,readonly=on',
               '-device', 'virtio-blk-device,drive=probe', '-cpu',
               'rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true']
    lockpath = os.environ['CAPSTONE_QEMU_LOCK']
    with open(lockpath, 'a+b') as lock, (scratch/'serial.log').open('wb') as log:
        if os.environ.get('CAPSTONE_QEMU_LOCK_HELD') != '1':
            fcntl.flock(lock, fcntl.LOCK_EX)
        guest = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=0)
        output=b''; login=False; sent=False
        deadline=time.monotonic()+180
        try:
            while time.monotonic()<deadline:
                if not select.select([guest.stdout], [], [], 1)[0]:
                    if guest.poll() is not None: break
                    continue
                chunk=os.read(guest.stdout.fileno(), 65536)
                if not chunk: break
                log.write(chunk); log.flush(); output+=chunk
                if not login and b'login:' in output:
                    guest.stdin.write(b'root\n'); guest.stdin.flush()
                    login=True; output=b''
                if login and not sent and b'# ' in output:
                    guest.stdin.write(b'mkdir -p /mnt/r0; mount -t ext4 -o ro /dev/vdb /mnt/r0 && '
                        b'insmod /mnt/r0/runtime_r0.ko && /mnt/r0/probe; rc=$?; '
                        b'rmmod runtime_r0; cleanup=$?; '
                        b'printf "R0_EXIT:%d CLEANUP:%d\\n" "$rc" "$cleanup"\n')
                    guest.stdin.flush(); sent=True; output=b''
                if sent and any(line.strip().startswith(b'R0_EXIT:')
                                for line in output.split(b'\n')[:-1]): break
                if b'Oops' in output or b'Kernel panic' in output or b'domain halted' in output: break
        finally:
            guest.terminate()
            try: guest.wait(timeout=5)
            except subprocess.TimeoutExpired: guest.kill(); guest.wait()
    lines=[line.strip() for line in output.splitlines()]
    names=['scattered_linux_mappings','cap_bounds','pte_write','cap_execute','pte_execute','fresh_retry','destroyed_context_tags_cleared','capstone_compiled_c']
    passed=all(lines.count(('R0:PASS '+name).encode())==1 for name in names)
    passed &= lines.count(b'VIRTUAL_RUNTIME_R0_OK tests=8')==1
    passed &= lines.count(b'R0_EXIT:0 CLEANUP:0')==1
    inputs={'qemu':a.qemu,'module':a.module,'probe':a.probe,'c_entry':a.c_entry,
            **{name:a.images/name for name in ('Image','fw_jump.elf','rootfs.ext2')}}
    result={'status':'PASS' if passed else 'FAIL', 'tests':names,
            'scope':'one hart, resident private mappings, explicit virtual C entry',
            'additional_kernel_core_patch':False, 'additional_firmware_patch':False,
            'control_omit_probe':a.omit_probe,
            'source_sha256':{path.name:digest(path) for path in sorted(HERE.iterdir())
                             if path.suffix in ('.c','.h','.S','.sh','.py')},
            'sha256':{name:digest(path) for name,path in inputs.items()}}
    a.record.parent.mkdir(parents=True,exist_ok=True)
    a.record.write_text(json.dumps(result,indent=2)+'\n')
    print(f"{result['status']}: serial log {scratch/'serial.log'}")
    for line in lines:
        if line.startswith((b'R0:',b'R0_FAIL:',b'VIRTUAL_RUNTIME_',b'R0_EXIT:')):
            print(line.decode(errors='replace'))
    return 0 if passed else 1

if __name__=='__main__':
    raise SystemExit(main())
