#!/usr/bin/env python3
"""Run the silicon-config SQLite domain on the FPGA from the BAKED initramfs copy.

Difference from run_sqlite_fpga.py: that one transfers the domain over the UART console.
At 1.6 MB that is tens of minutes even gzipped, and every transfer this project has run
today hit a sha mismatch and fell back to 1 char/emit. The domain is already inside the
firmware -- stage-sqlite-in-rootfs.sh puts it in the buildroot overlay, which becomes the
kernel's CONFIG_INITRAMFS_SOURCE cpio -- so it is present at /test-domains/ on every boot
with no transfer at all. The 17 MB firmware goes over JTAG in ~2 min regardless, so
carrying SQLite inside it is free.

That makes this a boot-and-run: no UART transfer, no sha retry ladder, no wedge risk from
a 1.6 MB type-in.

STALENESS IS THE HAZARD, and it is why this checks rather than assumes. A .dom frozen
inside an image survives rebuilds invisibly -- on 2026-07-29 the image carried a domain a
DAY older than the build, which would have re-measured already-fixed bugs. So this prints
the on-board size and compares it against the local build before running anything.

    usage: FPGA_URL=... FPGA_FW=<fw_payload.bin> python3 run_sqlite_baked_fpga.py
"""
import sys, os, pathlib, re, time

DRV = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(DRV.parent))
from fpga_driver import config as C
from fpga_driver.fpga_console import FpgaConsole
from fpga_driver.run_ladder_perf_fpga import cold_boot, nvbit, sh, install_resilient_emit

URL = os.environ.get("FPGA_URL")
IMG = pathlib.Path(os.environ["FPGA_FW"])
IMG_NAME = os.environ.get("FPGA_FW_NAME") or IMG.name
BITSTREAM = "working-caplifive-captype-fixed.bit"
TMP = pathlib.Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone"))
LOCAL_DOM = TMP / "sqlite-silicon" / "sqlite_silicon.dom"

DOM = "/test-domains/sqlite_silicon.dom"
HOST = "/test-domains/sqlite_host.user"

# The five markers run-sqlite-memory.sh gates on under QEMU. Same criterion here so a
# silicon pass means the same thing a QEMU pass does.
MARKERS = ("row name=alpha value=11", "row name=beta value=22",
           "row name=gamma value=33",
           "__CAPSTONE_SQLITE_EXTENDED_PASSED__",
           "__CAPSTONE_SQLITE_MEMORY_PASSED__")


def log(m): print(f"[sqlite] {m}", file=sys.stderr, flush=True)


def main():
    if not URL:
        raise SystemExit("FPGA_URL not set")
    if not IMG.is_file():
        raise SystemExit(f"missing firmware: {IMG}")
    local_size = LOCAL_DOM.stat().st_size if LOCAL_DOM.is_file() else -1
    log(f"local build: {LOCAL_DOM} = {local_size} bytes")

    console = FpgaConsole(URL, logger=lambda m: print(f"[fpga] {m}", file=sys.stderr))
    console.connect()
    install_resilient_emit(console)
    locked = False
    rc = 1
    try:
        console.lock(); locked = True
        rb = nvbit(console)
        if rb != BITSTREAM:
            raise SystemExit(f"HARD STOP: resident bitstream is {rb!r}, expected "
                             f"{BITSTREAM!r}. Re-flashing a shared board is never "
                             "automatic -- ask first.")
        log(f"took the lock; bitstream = {rb!r}")
        log(f"uploading firmware ({IMG.stat().st_size} bytes)")
        console.upload_boot_image(IMG_NAME, str(IMG))
        cold_boot(console, C.GDB_PROMPT)
        log("booted to a root shell with /dev/capstone present")

        # Prove the baked artifacts exist and match the local build BEFORE running.
        out = sh(console, f"ls -l {DOM} {HOST} 2>&1 | cat", timeout=30)
        print(out)
        m = re.search(rf"(\d+)\s+\S+\s+\S+\s+\S+\s+{re.escape(DOM)}", out) or \
            re.search(r"\s(\d{6,})\s", out)
        if m:
            on_board = int(m.group(1))
            log(f"on-board domain = {on_board} bytes; local = {local_size}")
            if local_size > 0 and on_board != local_size:
                log("WARNING: SIZE MISMATCH -- the image carries a STALE domain. "
                    "Re-run stage-sqlite-in-rootfs.sh and rebuild the firmware.")
        else:
            log("WARNING: could not read the on-board domain size from `ls`")

        log("running SQLite from the baked image (no transfer)")
        t0 = time.time()
        out = console.run_command(f"{HOST} {DOM}; echo D''N_$?", r"DN_\d", timeout=1800)
        log(f"run took {time.time()-t0:.0f}s")
        print(out)

        ok = True
        for mk in MARKERS:
            hit = mk in out
            ok &= hit
            print(f"  {'OK     ' if hit else 'MISSING'}  {mk}")
        cyc = re.search(r"mcycle[= ]+(\d+)", out)
        if cyc:
            print(f"  mcycle = {cyc.group(1)}")
        rc = 0 if ok else 1
        print("SQLITE ON SILICON: " + ("PASS" if ok else "FAIL"))
    finally:
        # Always leave the board powered off and unlocked, even on exception.
        try:
            console.power(False); log("powered off")
        except Exception as e:
            log(f"power off failed: {e}")
        try:
            if locked:
                console.unlock(); log("unlocked")
        except Exception as e:
            log(f"unlock failed: {e}")
        console.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
