#!/usr/bin/env python3
"""Run a staged application gate on the native Linux platform.

The kernel command line and the -cpu string start with the native platform's words
(native_platform.py). A later word wins in both, so --kernel-arg and
CAPSTONE_QEMU_CPU_EXTRA change a default without removing it.
"""
import argparse
import fcntl
import os
from pathlib import Path
import select
import subprocess
import time
import native_platform
from qemu_lock import qemu_lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qemu', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--firmware', type=Path, help='Override images/fw_jump.elf')
    p.add_argument('--dtb', type=Path, help='Override platform device tree (qualification fixtures)')
    p.add_argument('--stage', type=Path, required=True, help='Contains executable gate.sh and application resources')
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--disk-mib', type=int, default=1024)
    p.add_argument('--memory', default=native_platform.GUEST_MEMORY, help="Guest RAM, as QEMU's -m takes it")
    p.add_argument('--cpus', type=int, default=1, help='Guest CPUs (default: 1)')
    p.add_argument('--kernel-arg', action='append', default=[],
                   help='Additional kernel command-line argument (repeatable)')
    p.add_argument('--work', type=Path, required=True, help='New output directory')
    p.add_argument("--exact-bounds", action=argparse.BooleanOptionalAction, default=False,
                   help="Run QEMU's exact-bounds profile. Option (iii) does not need it; it is kept so a suite can be run both ways as a two-sided control.")
    p.add_argument('--share', type=Path,
                   help='Directory exported to the guest over 9p (mount tag gate9p), for gates that need it')
    p.add_argument('--plugin', type=Path, help='Optional QEMU TCG profiling plugin')
    p.add_argument('--plugin-arg', action='append', default=[])
    p.add_argument('--icount', type=int, choices=range(11))
    a = p.parse_args()
    if a.cpus < 1:
        p.error('--cpus must be positive')
    if a.plugin_arg and not a.plugin:
        p.error('--plugin-arg needs --plugin')
    if any(',' in arg or '=' not in arg for arg in a.plugin_arg):
        p.error('--plugin-arg must be key=value without commas')
    a.work.mkdir(parents=True, exist_ok=False)
    disk = a.work / 'stage.ext4'
    with disk.open('wb') as f:
        f.truncate(a.disk_mib << 20)
    subprocess.run(['/sbin/mkfs.ext4', '-F', '-q', '-d', str(a.stage), str(disk)], check=True)
    slots = native_platform.supervisor_slots_from_environment()
    cmd = [str(a.qemu), '-M', 'virt-capstone', '-m', a.memory, '-smp', str(a.cpus),
           '-nographic', '-monitor', 'none', '-serial', 'stdio', '-snapshot',
           '-bios', str(a.firmware if a.firmware else a.images/'fw_jump.elf'), '-kernel', str(a.images/'Image'),
           '-append', ' '.join(['root=/dev/vda ro', *native_platform.kernel_arguments(slots), *a.kernel_arg]),
           '-drive', f'file={a.images/"rootfs.ext2"},format=raw,id=hd0',
           '-device', 'virtio-blk-device,drive=hd0',
           '-drive', f'file={disk},format=raw,id=gate,readonly=on',
           '-device', 'virtio-blk-device,drive=gate', '-cpu',
           'rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true']
    cmd[-1] += ',' + ','.join(native_platform.cpu_properties(slots))
    if a.exact_bounds:
        cmd[-1] += ",x-capstone-exact-bounds=true"
    # CAPSTONE_QEMU_CPU_EXTRA appends to the -cpu string, so a reference pin differing only in a CPU
    # property (svadu=false, say) needs no wrapper script. A wrapper was the alternative and it is
    # worse for the RECORD: virtualvm.platform() hashes kit/qemu-system-riscv64, so a wrapper makes
    # the recorded "qemu" hash the wrapper's rather than the binary's, and a reader cannot tell
    # which QEMU produced the cells. Empty by default, so an unset variable cannot change a
    # recorded run -- checked by building the string both ways.
    #
    # The resolved string is written into the work directory as well as printed, because a flag
    # that is not in the artifacts is not evidence: platform() records file hashes and never the
    # command line, so without cpu.txt a reader cannot tell svadu=false from svadu default.
    cpu_extra = os.environ.get('CAPSTONE_QEMU_CPU_EXTRA', '').strip()
    if cpu_extra:
        cmd[-1] += "," + cpu_extra.lstrip(',')
    a.work.mkdir(parents=True, exist_ok=True)
    (a.work / 'cpu.txt').write_text(cmd[-1] + "\n")
    print(f"run-staged: -cpu {cmd[-1]}", flush=True)
    if a.plugin:
        cmd += ['-plugin', ','.join([str(a.plugin), *a.plugin_arg])]
    if a.dtb:
        cmd += ['-dtb', str(a.dtb)]
    if a.icount is not None:
        cmd += ['-icount', f'shift={a.icount}']
    if a.share:
        cmd += ['-virtfs', f'local,path={a.share.resolve()},mount_tag=gate9p,security_model=none,id=gate9p']
    completed = login = sent = False
    out = b''
    with qemu_lock(), (a.work/'serial.log').open('wb') as log:
        guest_env = dict(os.environ, TMPDIR=str(a.work.resolve()))
        guest = subprocess.Popen(cmd, env=guest_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=0)
        deadline = time.monotonic() + a.timeout
        try:
            while time.monotonic() < deadline:
                if not select.select([guest.stdout], [], [], 1)[0]:
                    if guest.poll() is not None:
                        break
                    continue
                data = os.read(guest.stdout.fileno(), 65536)
                if not data:
                    break
                log.write(data); log.flush()
                # Keep only the marker window; transcripts remain in the log.
                # Large network workloads must not repeatedly copy that log.
                out = (out + data)[-65536:]
                if not login and b'login:' in out:
                    guest.stdin.write(b'root\n'); guest.stdin.flush()
                    login = True; out = b''
                if login and not sent and b'# ' in out:
                    guest.stdin.write(b'mkdir -p /mnt/vm; mount -t ext4 -o ro /dev/vdb /mnt/vm && sh /mnt/vm/gate.sh\n')
                    guest.stdin.flush(); sent = True; out = b''
                if b'\nVIRTUAL_STAGED_DONE\r\n' in out:
                    completed = True
                    break
                if b'Kernel panic' in out or b'Oops' in out:
                    break
        finally:
            guest.terminate()
            try:
                guest.wait(timeout=5)
            except subprocess.TimeoutExpired:
                guest.kill(); guest.wait()
    print(a.work/'serial.log')
    # Completion only; each application gate must validate its own oracle.
    return 0 if completed else 1


if __name__ == '__main__':
    raise SystemExit(main())
