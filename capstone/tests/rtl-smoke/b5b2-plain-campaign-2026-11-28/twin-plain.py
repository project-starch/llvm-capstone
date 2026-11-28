#!/usr/bin/env python3
"""twin-plain.py IMGDIR OUTDIR GROUP...: the Plain arm's twin replay (runtime lane, 2026-11-28; PREREG.md).

Boots each group's Plain Image (IMGDIR/PLM-<group>/Image, its initramfs built in) on qemu-crv as a plain riscv64:
-cpu rv64,sstc=false with no Capstone property, 962M, the stock generic OpenSBI of the twins (sbi-keep-run fw-n,
keep-tags off), and captures the console to OUTDIR/PLM-<group>/boot.log. A functional replay and the hash
reference: the twin's time is not a prediction of the board's. Takes the shared QEMU lock once for all groups and
bounds each boot at 7,200 s (x264b at 14,400 s)."""
import os, select, subprocess, sys, time
H = os.environ['CAPSTONE_HDD']
sys.path.insert(0, H + '/c4/wt-zcb9o/capstone/runtime/virtual')
os.environ.setdefault('CAPSTONE_QEMU_LOCK', os.path.expanduser('~/.capstone-locks/qemu.lock'))
from qemu_lock import qemu_lock


def boot(img, out, bound):
    os.makedirs(out, exist_ok=True)
    cmd = [H + '/qemu-crv/build/qemu-system-riscv64', '-M', 'virt', '-m', '962M', '-smp', '1', '-nographic',
           '-cpu', 'rv64,sstc=false', '-bios', H + '/sbi-keep-run/fw-n/platform/generic/firmware/fw_jump.elf',
           '-kernel', img, '-append', 'console=ttyS0 earlycon=sbi rdinit=/init panic=-1']
    t0 = time.time()
    log = open(f'{out}/boot.log', 'wb')
    log.write(('# ' + ' '.join(cmd) + '\n').encode())
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    why = 'bound'
    while time.time() - t0 < bound:
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
    return why, rc, int(time.time() - t0)


def main():
    imgdir, outdir, groups = sys.argv[1], sys.argv[2], sys.argv[3:]
    print(time.strftime('%T'), 'waiting for the lock', flush=True)
    with qemu_lock():
        for g in groups:
            why, rc, s = boot(f'{imgdir}/PLM-{g}/Image', f'{outdir}/PLM-{g}', 14400 if g == 'x264b' else 7200)
            print(time.strftime('%T'), g, 'end', why, 'rc', rc, 'after', s, 's', flush=True)


main()
