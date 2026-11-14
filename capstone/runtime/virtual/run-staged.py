#!/usr/bin/env python3
"""Run a staged application gate on the trap-route platform (trap_platform.py).

One boot: QEMU virt with one hart and the trap-route CPU, the kit's busybox
initramfs as root, the stage directory as a read-only ext4 disk at /mnt/vm.
The initramfs /init runs /mnt/vm/gate.sh and powers off. No login, no module.

--images names a directory with Image, fw_jump.elf and initramfs.cpio.gz (a
kit's images/). Without initramfs.cpio.gz the boot builds one from the default
busybox base. --kernel-arg words follow the platform's own, and a later word
wins; capstone_vm.X words become sublet.X or drop out. CAPSTONE_QEMU_CPU_EXTRA
appends to the -cpu string, minus the deleted route's properties.
"""
import argparse
import os
from pathlib import Path

import trap_platform


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--qemu', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--firmware', type=Path, help='Override images/fw_jump.elf')
    p.add_argument('--dtb', type=Path, help='Override platform device tree (qualification fixtures)')
    p.add_argument('--stage', type=Path, required=True, help='Contains executable gate.sh and application resources')
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--disk-mib', type=int)
    p.add_argument('--memory', default=trap_platform.GUEST_MEMORY, help="Guest RAM, as QEMU's -m takes it")
    p.add_argument('--cpus', type=int, default=1, help='Guest CPUs: the trap route runs one hart')
    p.add_argument('--kernel-arg', action='append', default=[],
                   help='Additional kernel command-line argument (repeatable)')
    p.add_argument('--work', type=Path, required=True, help='New output directory')
    p.add_argument('--exact-bounds', action=argparse.BooleanOptionalAction, default=False,
                   help="Run QEMU's exact-bounds profile")
    p.add_argument('--share', type=Path,
                   help='Directory exported to the guest over 9p (mount tag gate9p), for gates that need it')
    p.add_argument('--plugin', type=Path, help='Optional QEMU TCG profiling plugin')
    p.add_argument('--plugin-arg', action='append', default=[])
    p.add_argument('--icount', type=int, choices=range(11))
    p.add_argument('--step-limit', type=int,
                   help="the gate's per-step watchdog bound, s: the host stops a guest silent for longer (a hang)")
    a = p.parse_args()

    if a.cpus != 1:
        p.error('--cpus: the trap route runs one hart')
    if a.plugin_arg and not a.plugin:
        p.error('--plugin-arg needs --plugin')
    if any(',' in arg or '=' not in arg for arg in a.plugin_arg):
        p.error('--plugin-arg must be key=value without commas')

    cpu_extra = 'x-capstone-exact-bounds=true' if a.exact_bounds else ''
    if cpu_extra:
        os.environ['CAPSTONE_QEMU_CPU_EXTRA'] = ','.join(
            w for w in (os.environ.get('CAPSTONE_QEMU_CPU_EXTRA', '').strip(','), cpu_extra) if w)

    completed = trap_platform.boot(a.qemu, a.images, a.stage, a.work, a.timeout, a.kernel_arg, a.disk_mib,
                                   a.memory, a.firmware, share=a.share, plugin=a.plugin, plugin_args=a.plugin_arg,
                                   icount=a.icount, dtb=a.dtb, log_prefix='run-staged',
                                   step_limit=a.step_limit)
    print(a.work / 'serial.log')
    # Completion only; each application gate must validate its own oracle.
    return 0 if completed else 1


if __name__ == '__main__':
    raise SystemExit(main())
