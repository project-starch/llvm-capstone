#!/usr/bin/env python3
"""twin-dry.py OUTDIR: boot this kit's Image on the qemu-crv twin exactly as the campaign twins boot (camp-twin.sh,
perlcount/count.py: 962M, fw-y fw_jump, the same cpu properties), the overlay's initramfs as -initrd, and capture the
console to OUTDIR/boot.log. A dry run of the instrument, decode and reader: the twin's cycle CSR does not run at
25 MHz and its hpmcounter3 counts nothing, so the clock and mcycle controls are expected to FAIL here. Holds the
shared QEMU lock for the boot only; bounded at 5,400 s."""
import os, subprocess, sys, time, select
H = os.environ['CAPSTONE_HDD']  # the shared HDD root, as capstone-test-env.sh exports it
sys.path.insert(0, H + '/c4/wt-zcb9o/capstone/runtime/virtual')
os.environ.setdefault('CAPSTONE_QEMU_LOCK', os.path.expanduser('~/.capstone-locks/qemu.lock'))
from qemu_lock import qemu_lock


def main():
    out = sys.argv[1]
    T = H + '/ticklog'
    os.makedirs(out, exist_ok=True)
    cpu = 'rv64,sstc=false,x-capstone-trap-route=true,x-capstone-supervisor-refs=true,x-capstone-raw-moves=true'
    cmd = [H + '/qemu-crv/build/qemu-system-riscv64', '-M', 'virt', '-m', '962M', '-smp', '1',
           '-nographic', '-cpu', cpu,
           '-bios', H + '/sbi-keep-run/fw-y/platform/generic/firmware/fw_jump.elf',
           '-kernel', f'{T}/out/Image', '-initrd', f'{T}/out/initramfs.cpio.gz',
           '-append', 'console=ttyS0 earlycon=sbi rdinit=/init panic=-1 sublet.dom_binfmt=1 ']
    print(time.strftime('%T'), 'waiting for the lock', flush=True)
    with qemu_lock():
        t0 = time.time()
        log = open(f'{out}/boot.log', 'wb')
        log.write(('# ' + ' '.join(cmd) + '\n').encode())
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        why = 'bound'
        while time.time() - t0 < 5400:
            r, _, _ = select.select([p.stdout], [], [], 2)
            if r:
                d = os.read(p.stdout.fileno(), 65536)
                if not d:
                    why = 'exited'
                    break
                log.write(d)
                log.flush()
        if p.poll() is None:
            p.kill()
        rc = p.wait()
        log.close()
        print(time.strftime('%T'), 'end', why, 'rc', rc, 'after', int(time.time() - t0), 's', flush=True)


main()
