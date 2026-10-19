#!/usr/bin/env python3
"""Exec in place and the image's write denial. The launcher takes no snapshot of
an image on a filesystem with one inode per file: the module denies writers to
the file (ETXTBSY) from before the image is checked until the process ends. A
memfd must be sealed; on other filesystems and for exec in place the image is a
sealed copy. Checked from the guest: exec in place (same pid), exec of a
truncated image (ENOEXEC, in the caller) and of a native program (ENOSYS),
writing a running image (denied) and the same image after exit (allowed), a
launch while a writer is open (refused) and after it closed, memfds in three
seal states, images on a ramfs and on 9p (both run from a sealed copy: the
launcher's image descriptor is checked on 9p), and on ext4 over a loop device
(the denial on a block filesystem: a writer through a hard link and a shared
writable mapping older than the launch are refused, and CV_IMAGE a second time
answers EBUSY). The other images run from /tmp, a tmpfs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
SCRIPT = r'''#!/bin/sh
dmesg -n 1
insmod /mnt/vm/capstone_vm.ko || { echo "@@INSMOD failed"; echo VIRTUAL_STAGED_DONE; exit 1; }
V=/mnt/vm/capstone-vexec
cd /tmp
cp /mnt/vm/exec.dom /tmp/e.dom && chmod 755 /tmp/e.dom
$V /tmp/e.dom exec /tmp/e.dom > o1 2>&1; echo "@@EXEC rc=$? $(tr '\n' ' ' < o1)"
dd if=/mnt/vm/exec.dom of=/tmp/trunc.dom bs=4096 count=1 2>/dev/null && chmod 755 /tmp/trunc.dom
$V /tmp/e.dom refused /tmp/trunc.dom > o2 2>&1; echo "@@REFUSED rc=$? $(tr '\n' ' ' < o2)"
echo junk > /tmp/notimage && chmod 755 /tmp/notimage
$V /tmp/e.dom refused /tmp/notimage > o2b 2>&1; echo "@@REFUSED_NATIVE rc=$? $(tr '\n' ' ' < o2b)"
mkfifo /tmp/release
$V /tmp/e.dom hold < /tmp/release > o3 2>&1 &
pid=$!
exec 8> /tmp/release
n=0; while ! grep -q EXEC_CONTRACT_HOLDING o3 && [ $n -lt 30 ]; do sleep 1; n=$((n+1)); done
if ! grep -q EXEC_CONTRACT_HOLDING o3; then echo "@@WRITE_WHILE_RUNNING not-ready"
elif (echo x >> /tmp/e.dom) 2>/dev/null; then echo "@@WRITE_WHILE_RUNNING allowed"
else echo "@@WRITE_WHILE_RUNNING denied"; fi
exec 8>&-
wait $pid; echo "@@HOLD rc=$? $(tr '\n' ' ' < o3)"
if (echo x >> /tmp/e.dom) 2>/dev/null; then echo "@@WRITE_AFTER_EXIT allowed"; else echo "@@WRITE_AFTER_EXIT denied"; fi
cp /mnt/vm/exec.dom /tmp/f.dom && chmod 755 /tmp/f.dom
exec 7>>/tmp/f.dom
$V /tmp/f.dom child > o5 2>&1; echo "@@OPEN_WRITER rc=$? $(tr '\n' ' ' < o5)"
exec 7>&-
$V /tmp/f.dom child > o6 2>&1; echo "@@AFTER_WRITER rc=$? $(tr '\n' ' ' < o6)"
# ext4 on a loop device: the denial on a block filesystem with one inode per
# file (the arms above run from tmpfs), a writer through a second hard link, a
# shared writable mapping older than the launch, and the module asked twice.
dd if=/dev/zero of=/tmp/e4.img bs=1M count=8 2>/dev/null && mke2fs -F /tmp/e4.img >/dev/null 2>&1 && mkdir -p /mnt/e4 && mount -t ext4 -o loop /tmp/e4.img /mnt/e4
echo "@@EXT4_FS $(sed -n 's/^[^ ]* \/mnt\/e4 \([^ ]*\) .*/\1/p' /proc/mounts)"
cp /mnt/vm/exec.dom /mnt/e4/e.dom && chmod 755 /mnt/e4/e.dom && ln /mnt/e4/e.dom /mnt/e4/link.dom
mkfifo /tmp/release4
$V /mnt/e4/e.dom hold < /tmp/release4 > o4 2>&1 &
pid=$!
exec 5> /tmp/release4
n=0; while ! grep -q EXEC_CONTRACT_HOLDING o4 && [ $n -lt 30 ]; do sleep 1; n=$((n+1)); done
if ! grep -q EXEC_CONTRACT_HOLDING o4; then echo "@@EXT4_WRITE not-ready"
elif (echo x >> /mnt/e4/e.dom) 2>/dev/null; then echo "@@EXT4_WRITE allowed"
else echo "@@EXT4_WRITE denied"; fi
if (echo x >> /mnt/e4/link.dom) 2>o4l; then echo "@@EXT4_LINK_WRITE allowed"; else echo "@@EXT4_LINK_WRITE denied $(tr '\n' ' ' < o4l)"; fi
exec 5>&-
wait $pid; echo "@@EXT4_HOLD rc=$? $(tr '\n' ' ' < o4)"
if (echo x >> /mnt/e4/link.dom) 2>/dev/null; then echo "@@EXT4_LINK_AFTER allowed"; else echo "@@EXT4_LINK_AFTER denied"; fi
cp /mnt/vm/exec.dom /mnt/e4/m.dom && chmod 755 /mnt/e4/m.dom
/mnt/vm/memfd-launch mapped $V /mnt/e4/m.dom > o4m 2>&1; echo "@@MAPPED_WRITER rc=$? $(tr '\n' ' ' < o4m)"
$V /mnt/e4/m.dom child > o4n 2>&1; echo "@@AFTER_MAPPING rc=$? $(tr '\n' ' ' < o4n)"
/mnt/vm/memfd-launch image-twice /mnt/e4/m.dom > o4t 2>&1; echo "@@IMAGE_TWICE rc=$? $(tr '\n' ' ' < o4t)"
mkdir -p /tmp/r && mount -t ramfs ramfs /tmp/r && cp /mnt/vm/exec.dom /tmp/r/e.dom && chmod 755 /tmp/r/e.dom
exec 9>>/tmp/r/e.dom
$V /tmp/r/e.dom child > o8 2>&1; echo "@@COPY_FS rc=$? $(tr '\n' ' ' < o8)"
exec 9>&-
mkdir -p /mnt/9p
if mount -t 9p -o trans=virtio,version=9p2000.L gate9p /mnt/9p; then
  # 9p (cache=none) makes a new inode at each lookup: the module refuses it
  # and the launcher holds a sealed copy, not the 9p file.
  mkfifo /tmp/release9
  $V /mnt/9p/exec.dom hold < /tmp/release9 > o9 2>&1 &
  pid=$!
  exec 6> /tmp/release9
  n=0; while ! grep -q EXEC_CONTRACT_HOLDING o9 && [ $n -lt 30 ]; do sleep 1; n=$((n+1)); done
  fds=$(ls -l /proc/$pid/fd 2>/dev/null)
  echo "@@NINEP_FDS memfd=$(echo "$fds" | grep -c 'memfd:capstone-application') share=$(echo "$fds" | grep -c '/mnt/9p/')"
  exec 6>&-
  wait $pid; echo "@@NINEP rc=$? $(tr '\n' ' ' < o9)"
else
  echo "@@NINEP_FDS mount-failed"
fi
for mode in open write sealed; do
  /mnt/vm/memfd-launch $mode $V /mnt/vm/exec.dom > o7 2>&1; echo "@@MEMFD_$mode rc=$? $(tr '\n' ' ' < o7)"
done
$V --stats
rmmod capstone_vm
echo VIRTUAL_STAGED_DONE
'''
CHILD = 'EXEC_CONTRACT_CHILD'
BUSY = 'Text file busy'
ENOEXEC, ENOSYS = 8, 38


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qemu', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--application', type=Path, required=True, help='exec-contract.c built with the SDK')
    p.add_argument('--helper', type=Path, required=True, help='memfd-launch.c built static for the guest')
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'; stage.mkdir()
    inputs = {'qemu': a.qemu, 'application': a.application, 'helper': a.helper,
              'launcher': a.adapter/'capstone-vexec',
              'module': a.adapter/'module/capstone_vm.ko',
              **{n: a.images/n for n in ('Image', 'fw_jump.elf', 'rootfs.ext2')}}
    hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    for src, name in ((inputs['application'], 'exec.dom'), (inputs['helper'], 'memfd-launch'),
                      (inputs['launcher'], 'capstone-vexec'),
                      (inputs['module'], 'capstone_vm.ko')):
        shutil.copy2(src, stage/name)
    (stage/'gate.sh').write_text(SCRIPT)
    share = a.work / 'share'; share.mkdir()
    shutil.copy2(inputs['application'], share/'exec.dom')
    run = subprocess.run([os.environ.get('PYTHON', 'python3'),
        str(HERE/'run-staged.py'), '--qemu', str(a.qemu), '--images', str(a.images),
        '--stage', str(stage), '--work', str(a.work/'guest'), '--disk-mib', '64',
        '--timeout', str(a.timeout), '--share', str(share)])
    log = (a.work/'guest/serial.log').read_text(errors='replace').replace('\r', '')
    lines = {m.group(1): (m.group(2), m.group(3)) for m in
             re.finditer(r'^@@(\w+)(?: rc=(\d+))? ?(.*)$', log, re.M)}

    def rc(name):
        v = lines.get(name)
        return int(v[0]) if v and v[0] is not None else None

    def text(name):
        return lines.get(name, (None, ''))[1]

    census = [json.loads(l) for l in log.splitlines() if l.startswith('{"version":1,')]
    checks = {
        'completed': run.returncode == 0 and 'VIRTUAL_STAGED_DONE' in log,
        # Same process: the child reports the parent's pid.
        'exec_in_place': rc('EXEC') == 0 and bool(re.search(
            r'EXEC_CONTRACT_PARENT pid=(\d+) .*EXEC_CONTRACT_CHILD pid=\1\b', text('EXEC'))),
        # A truncated image is an image that fails the check: ENOEXEC, in the caller.
        'exec_refused_returns_enoexec': rc('REFUSED') == 0 and
            f'EXEC_CONTRACT_REFUSED rc=-1 errno={ENOEXEC} ' in text('REFUSED') + ' ',
        # A native program cannot replace a filtered task: ENOSYS (applications.md).
        'exec_native_returns_enosys': rc('REFUSED_NATIVE') == 0 and
            f'EXEC_CONTRACT_REFUSED rc=-1 errno={ENOSYS} ' in text('REFUSED_NATIVE') + ' ',
        'running_image_not_writable': text('WRITE_WHILE_RUNNING') == 'denied',
        'holder_unaffected': rc('HOLD') == 0 and 'EXEC_CONTRACT_HELD' in text('HOLD'),
        'writable_after_exit': text('WRITE_AFTER_EXIT') == 'allowed',
        'open_writer_refused': rc('OPEN_WRITER') == 126 and BUSY in text('OPEN_WRITER') and
            CHILD not in text('OPEN_WRITER'),
        'runs_after_writer_closed': rc('AFTER_WRITER') == 0 and CHILD in text('AFTER_WRITER'),
        # ext4 on a loop device: the denial on a block filesystem, which the
        # tmpfs arms above never exercise. A hard link names the same inode,
        # so a writer through it is counted and refused with ETXTBSY.
        'ext4_mounted': text('EXT4_FS') == 'ext4',
        'ext4_running_image_not_writable': text('EXT4_WRITE') == 'denied',
        'ext4_hard_link_writer_refused': text('EXT4_LINK_WRITE').startswith('denied') and
            BUSY in text('EXT4_LINK_WRITE'),
        'ext4_holder_unaffected': rc('EXT4_HOLD') == 0 and 'EXEC_CONTRACT_HELD' in text('EXT4_HOLD'),
        'ext4_link_writable_after_exit': text('EXT4_LINK_AFTER') == 'allowed',
        # A shared writable mapping made before the launch keeps the file open
        # for writing after its descriptor closed: the launch is refused, and
        # runs once the mapping holder is gone.
        'mapped_writer_refused': rc('MAPPED_WRITER') == 126 and BUSY in text('MAPPED_WRITER') and
            CHILD not in text('MAPPED_WRITER'),
        'runs_after_mapping_gone': rc('AFTER_MAPPING') == 0 and CHILD in text('AFTER_MAPPING'),
        # CV_IMAGE twice on one context: EBUSY, and the first denial stands
        # until the device is closed.
        'image_twice_busy': rc('IMAGE_TWICE') == 0 and
            'IMAGE_TWICE first=ok second=EBUSY held=ETXTBSY released=ok' in text('IMAGE_TWICE'),
        'memfd_unsealed_refused': rc('MEMFD_open') == 126 and BUSY in text('MEMFD_open') and
            CHILD not in text('MEMFD_open'),
        'memfd_write_sealed_only_refused': rc('MEMFD_write') == 126 and BUSY in text('MEMFD_write') and
            CHILD not in text('MEMFD_write'),
        'memfd_sealed_runs': rc('MEMFD_sealed') == 0 and CHILD in text('MEMFD_sealed'),
        # ramfs is not on the launcher's list of one-inode-per-file filesystems:
        # the image runs from a sealed copy, so a writer open on the file does
        # not stop it (a direct load would fail with ETXTBSY).
        'other_filesystem_runs_from_copy': rc('COPY_FS') == 0 and CHILD in text('COPY_FS'),
        # 9p: the launcher's image descriptor is the sealed copy, not the 9p file.
        'ninep_runs_from_copy': rc('NINEP') == 0 and 'EXEC_CONTRACT_HELD' in text('NINEP') and
            text('NINEP_FDS') == 'memfd=1 share=0',
        'cleanup': len(census) == 1 and all(
            s['live_domains'] == s['live_regions'] == s['live_bytes'] == 0 for s in census),
    }
    checks['inputs_unchanged'] = hashes == {
        k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()}
    result = {'schema': 1, 'status': 'PASS' if all(checks.values()) else 'FAIL',
              'checks': checks, 'lines': {k: {'rc': v[0], 'text': v[1]} for k, v in lines.items()},
              'sha256': hashes,
              'source_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), HERE/'exec-contract.c',
                                          HERE/'memfd-launch.c', HERE/'run-staged.py']},
              'gate_sha256': hashlib.sha256(SCRIPT.encode()).hexdigest(),
              # The 9p check catches a launcher that trusts statfs() only where the
              # share's host filesystem would pass such a check (ext4, tmpfs, ...).
              'share_host_filesystem': subprocess.run(['stat', '-f', '-c', '%T', str(share)],
                                                      capture_output=True, text=True).stdout.strip(),
              'scope': 'one hart; images on tmpfs, on ext4 over a loop device, in memfds, on ramfs and on 9p (sealed copy); exec in place of a valid and a non-image file'}
    (a.work/'result.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, ok in checks.items(): print(('PASS ' if ok else 'FAIL ') + name)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
