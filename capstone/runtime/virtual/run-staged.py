#!/usr/bin/env python3
"""Run a staged application gate on the qualified virtual Linux platform."""
import argparse
import json
import fcntl
import os
from pathlib import Path
import select
import subprocess
import time
from qemu_lock import qemu_lock
from host_window import HostWindow
from host_profile import HostProfile


def prepare_share(share, private, domain):
    """Reserve private output atomically; shared directories stay exclusive."""
    if private and share is None:
        raise ValueError('--private-share requires --share')
    if domain and share is not None and not private:
        raise ValueError('A QEMU lock domain requires --private-share for 9p output')
    if private:
        # mkdir without exist_ok also refuses existing empty directories and
        # symlinks. Two runners cannot both reserve the same output path.
        share.mkdir(mode=0o700)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qemu', type=Path, required=True)
    p.add_argument('--machine', default='virt-capstone', choices=['virt', 'virt-capstone'])
    p.add_argument('--cpu', default='rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true')
    p.add_argument('--host-window', action='store_true',
                   help='Measure host wall/CPU time between HOST_WINDOW_BEGIN/END serial lines; no plugins')
    p.add_argument('--host-perf', type=Path,
                   help='Diagnostic only: perf binary inside the host window; excludes clean timing')
    p.add_argument('--host-perf-mode', choices=['record', 'stat'], default='record',
                   help='Sample stacks or count host user-mode instructions/cycles/branches')
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--firmware', type=Path, help='Override images/fw_jump.elf')
    p.add_argument('--dtb', type=Path, help='Override platform device tree (qualification fixtures)')
    p.add_argument('--stage', type=Path, required=True, help='Contains executable gate.sh and application resources')
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--disk-mib', type=int, default=1024)
    p.add_argument('--memory', default='2G', help="Guest RAM, as QEMU's -m takes it")
    p.add_argument('--cpus', type=int, default=1, help='Guest CPUs (default: 1)')
    p.add_argument('--kernel-arg', action='append', default=[],
                   help='Additional kernel command-line argument (repeatable)')
    p.add_argument('--work', type=Path, required=True, help='New output directory')
    p.add_argument("--exact-bounds", action=argparse.BooleanOptionalAction, default=False,
                   help="Run QEMU's exact-bounds profile. Option (iii) does not need it; it is kept so a suite can be run both ways as a two-sided control.")
    p.add_argument('--share', type=Path,
                   help='Directory exported to the guest over 9p (mount tag gate9p), for gates that need it')
    p.add_argument('--private-share', action='store_true',
                   help='Create a new private --share directory; required for 9p in a QEMU lock domain')
    p.add_argument('--plugin', type=Path, help='Optional QEMU TCG profiling plugin')
    p.add_argument('--plugin-arg', action='append', default=[])
    p.add_argument('--icount', type=int, choices=range(11))
    a = p.parse_args()
    if a.cpus < 1:
        p.error('--cpus must be positive')
    if a.plugin_arg and not a.plugin:
        p.error('--plugin-arg needs --plugin')
    if a.host_window and a.plugin:
        p.error('--host-window timing must not include a profiling plugin')
    if a.host_perf and not a.host_window:
        p.error('--host-perf requires --host-window')
    if a.host_perf_mode != 'record' and not a.host_perf:
        p.error('--host-perf-mode stat requires --host-perf')
    domain = os.environ.get('CAPSTONE_QEMU_DOMAIN_LOCK', '').strip()
    if a.host_window and domain:
        p.error('Host measurement windows require the global exclusive QEMU lock')
    if any(',' in arg or '=' not in arg for arg in a.plugin_arg):
        p.error('--plugin-arg must be key=value without commas')
    a.work.mkdir(parents=True, exist_ok=False)
    try:
        prepare_share(a.share, a.private_share, domain)
    except (ValueError, OSError) as error:
        p.error(str(error))
    isolation = dict(
        snapshot=True, private_share=a.private_share,
        domain_lock=domain or None, host_affinity=sorted(os.sched_getaffinity(0)),
        timing_scope='Concurrent functional validation only' if domain else 'Global exclusive',
    )
    (a.work/'isolation.json').write_text(json.dumps(isolation, indent=2)+'\n')
    disk = a.work / 'stage.ext4'
    with disk.open('wb') as f:
        f.truncate(a.disk_mib << 20)
    subprocess.run(['/sbin/mkfs.ext4', '-F', '-q', '-d', str(a.stage), str(disk)], check=True)
    cmd = [str(a.qemu), '-M', a.machine, '-m', a.memory, '-smp', str(a.cpus),
           '-nographic', '-monitor', 'none', '-serial', 'stdio', '-snapshot',
           '-bios', str(a.firmware if a.firmware else a.images/'fw_jump.elf'), '-kernel', str(a.images/'Image'),
           '-append', ' '.join(['root=/dev/vda ro', *a.kernel_arg]),
           '-drive', f'file={a.images/"rootfs.ext2"},format=raw,id=hd0',
           '-device', 'virtio-blk-device,drive=hd0',
           '-drive', f'file={disk},format=raw,id=gate,readonly=on',
           '-device', 'virtio-blk-device,drive=gate', '-cpu',
           a.cpu]
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
    if a.host_perf and a.host_perf_mode == 'record':
        cmd += ['-perfmap']
    (a.work/'qemu-command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    completed = login = sent = False
    out = b''
    with qemu_lock(), (a.work/'serial.log').open('wb') as log:
        guest_env = dict(os.environ, TMPDIR=str(a.work.resolve()))
        guest = subprocess.Popen(cmd, env=guest_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=0)
        isolation.update(guest_pid=guest.pid, started_monotonic_ns=time.monotonic_ns())
        window = HostWindow(guest.pid) if a.host_window else None
        profile = HostProfile(a.host_perf, a.work, guest.pid, a.host_perf_mode) if a.host_perf else None
        profile_started = False
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
                if window:
                    window.feed(data)
                if profile and window.begin and not profile_started:
                    profile.start()
                    profile_started = True
                if profile and window.end:
                    profile.stop()
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
            isolation['ended_monotonic_ns'] = time.monotonic_ns()
            (a.work/'isolation.json').write_text(json.dumps(isolation, indent=2)+'\n')
            if profile:
                profile.stop()
                profile.save_map()
        if window and completed:
            result = window.result()
            result['profiled'] = bool(profile)
            if profile:
                result['profile_mode'] = a.host_perf_mode
            (a.work/'host-window.json').write_text(json.dumps(result, indent=2)+'\n')
    print(a.work/'serial.log')
    # Completion only; each application gate must validate its own oracle.
    return 0 if completed else 1


if __name__ == '__main__':
    raise SystemExit(main())
