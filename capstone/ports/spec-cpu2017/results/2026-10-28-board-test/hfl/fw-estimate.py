#!/usr/bin/env python3
"""fw-estimate.py [--cap BYTES] STAGE...: the board firmware size a make-stage.py stage will bake to, from its gzip size.

The board kit bakes the stage into the Linux Image's initramfs (a gzip'd newc cpio) and the firmware build appends the
Image to OpenSBI; board-run.sh refuses a firmware over BOARD_FW_CAP before the boot. Measured on six past boots of the
same kit (bkit12-perf-b3, base manifest bkit7-rev2xDN2), by extracting the initramfs from each Image:
  - initramfs gzip length = BASE_GZ + gzip -9 of the stage's tar, within 30 KB on four stages from 0.26 to 6.4 MB
    (lbm test 3,700,441 measured; imagick train, the reg_email v2 stage and xz test predicted to -6, -28 and -3 KB);
  - the initramfs starts at 6,349,144 in every Image;
  - the Image size is one of 15,370,752 + k x 2 MiB, the smallest that holds 6,349,144 + initramfs + TAIL, where TAIL
    lies in (4,095,735, 5,321,167] (bounded by the lbm, xz-test, imagick-train and reg_email Images);
  - the firmware is the Image + 2,097,160 B (every one of them).
Prints the firmware range for the two ends of TAIL and the verdict against the cap: FITS only if the upper end fits.
Exit 1 if any stage's upper end is over the cap, 2 on no data.
"""
import argparse
import subprocess
import sys
from pathlib import Path

BASE_GZ = 3_700_441 - 259_776          # lbm test's initramfs minus the gzip -9 of its stage (stages/519.lbm_r-test)
OFF, G0, STEP, FW_EXTRA = 6_349_144, 15_370_752, 2 * 1024 * 1024, 2_097_160
TAIL = (4_095_736, 5_321_167)


def image(end):
    k = max(0, -(-(end - G0) // STEP))
    return G0 + k * STEP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=26_000_000)
    ap.add_argument("stages", nargs="+", type=Path)
    a = ap.parse_args()
    rc = 0
    for st in a.stages:
        d = st / "stage" if (st / "stage").is_dir() else st
        if not d.is_dir():
            print(f"NO DATA: {st}")
            return 2
        gz = int(subprocess.run(f"tar -C '{d.parent}' -cf - '{d.name}' | gzip -9 | wc -c", shell=True, check=True,
                                capture_output=True, text=True).stdout)
        lo, hi = (image(OFF + BASE_GZ + gz + t) + FW_EXTRA for t in TAIL)
        ok = hi <= a.cap
        rc = max(rc, 0 if ok else 1)
        print(f"{st.name}: stage gzip {gz:,} B -> firmware {lo:,}..{hi:,} B, cap {a.cap:,}: "
              f"{'FITS' if ok else ('OVER' if lo > a.cap else 'MAY NOT FIT')}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
