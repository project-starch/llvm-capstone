#!/usr/bin/env python3
"""DEPRECATED 2026-08-03 -- this driver ships the program to the board over the UART
console. Do not use it for new work, and do not extend it.

Use the BAKED path instead: put the artifact in
`caplifive-system/sw/buildroot/overlay/test-domains/` AND `build/target/test-domains/`,
rebuild with `A=linux-rebuild` then `A=opensbi-rebuild`, and invoke it from the shell
with `run_baked_rungs_fpga.py` (ladder rungs) or `run_sqlite_stages_fpga.py` (SQLite /
staged probes). Rationale: UART delivery is 16 chars per socket.io emit and each emit is
an HTTPS round trip, so a ~10 KB domain costs MINUTES; baked, 10 domains run in ONE boot
in ~5 minutes because the JTAG upload happens anyway. See
`agent-handoff/ref/HOW-TO-LAUNCH-ON-FPGA.md` s"UART TRANSFER IS RETIRED".
"""
"""Run the silicon-config SQLite domain on the FPGA board and read mcycle.

DELIVERY: gzip+base64 over the UART console, via fast_put. That is a deliberate choice
over JTAG, and the reasoning is worth stating because it was got wrong once:

  * `monitor load_image` writes to a PHYSICAL RAM ADDRESS -- no filesystem involved --
    so using it for the domain means choosing an address. Guessing one is precisely
    what the board rules forbid (it stomps the running kernel). A safe address needs
    either the board owner's confirmation or memory reserved via bootargs/DTB, which
    is itself a firmware change.
  * Baking the domain into the image (the other JTAG route) needs a KERNEL rebuild:
    the FPGA rootfs is a CPIO initramfs compiled into the kernel
    (CONFIG_INITRAMFS_SOURCE), and that buildroot tree has never been built here.
  * UART was initially ruled out at ">= 63 min", which was WRONG: that computed base64
    of the RAW image, but fast_put gzips first. SQLite compresses 1.62 MB -> 529 KB,
    so it is ~705 K base64 chars, ~15 min at the 800 chars/s pacing floor. Acceptable
    for a one-off measurement.

If many board iterations are ever needed, revisit JTAG -- but with a board-owner-
confirmed reserved region, not a guessed address.
"""
import os, sys, pathlib, time, subprocess

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fpga_driver.fpga_console import FpgaConsole
from fpga_driver.fast_xfer import fast_put
import fpga_driver.config as C
# Reuse the ladder runner's boot sequence rather than reimplementing it. cold_boot
# encodes several hard-won details -- a warm `monitor reset halt` does NOT work here
# (the fw_payload's one-time hart/DDR init is not re-runnable, so a real power cycle
# is required), the pc/a0 must be set explicitly before `continue`, and the initramfs
# is re-unpacked on every boot so /tmp starts empty. Writing that again from the API
# docs is how the first attempt failed.
from fpga_driver.run_ladder_perf_fpga import cold_boot, ensure_capstone_dev, nvbit

URL = os.environ.get("FPGA_URL")
IMG_NAME = "fw_payload_fpga_up_gpfree.bin"
IMG = pathlib.Path(os.environ.get("FPGA_FW",
      str(pathlib.Path.home() / "capstone-b-artifacts" / IMG_NAME)))
TMP = pathlib.Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone"))
DOM = pathlib.Path(os.environ.get("SQLITE_DOM", TMP / "sqlite-silicon" / "sqlite_silicon.dom"))
HOST = pathlib.Path(os.environ.get("SQLITE_HOST", TMP / "sqlite-build" / "sqlite_host.user"))
# Resident-bitstream guard. Reflashed 2026-08-12 to caplifive_12august.bit, which adds the
# latched-mepc debug mux and the TOTAL LCC field-1 type query on top of the 2026-08-04
# operand-forwarding fix (capstone-ariane 7aac52f93). Overridable so the next reflash needs
# no code change mid-session -- the guard exists to stop a run from silently measuring the
# WRONG silicon, and on 2026-08-04 it did exactly that.
#
# KEEP THIS DEFAULT IN STEP WITH WHAT IS ACTUALLY FLASHED. It was left naming the 08-04
# bitstream for ten days after the 08-12 reflash, so every boot in between had to pass
# FPGA_BITSTREAM by hand and one that forgot burned a launch on a HARD STOP. A default that
# is always overridden trains people to override it, which is how a real mismatch gets waved
# through.
BITSTREAM = os.environ.get("FPGA_BITSTREAM", "caplifive_12august.bit")


def log(m): print(f"[sqlite] {m}", file=sys.stderr, flush=True)


def gz(path):
    out = TMP / (path.name + ".gz")
    subprocess.run(["gzip", "-9", "-c", str(path)], stdout=open(out, "wb"), check=True)
    # fast_put verifies the DECOMPRESSED file on the board against this, first 16 hex.
    sha = subprocess.run(["sha256sum", str(path)], capture_output=True,
                         text=True).stdout.split()[0][:16]
    return out, sha


def main():
    if not URL:
        raise SystemExit("FPGA_URL not set")
    for p in (IMG, DOM, HOST):
        if not p.is_file():
            raise SystemExit(f"missing artifact: {p}")

    console = FpgaConsole(URL, logger=lambda m: print(f"[fpga] {m}", file=sys.stderr))
    console.connect()
    locked = False
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

        for src, dst in ((HOST, "/tmp/sqlite_host.user"), (DOM, "/tmp/sqlite.dom")):
            g, sha = gz(src)
            log(f"transferring {src.name}: {src.stat().st_size} B raw -> "
                f"{g.stat().st_size} B gzipped. Minutes, not seconds.")
            t0 = time.time()
            fast_put(console, str(g), dst + ".gz", dst, sha,
                     do_exec=dst.endswith(".user"), log=log)
            log(f"  {src.name} took {time.time()-t0:.0f}s")

        log("running SQLite in the domain")
        out = console.run_command("/tmp/sqlite_host.user /tmp/sqlite.dom; echo D''N_$?",
                                  r"DN_\d", timeout=1800)
        print(out)
        ok = True
        for m in ("row name=alpha value=11", "row name=beta value=22",
                  "row name=gamma value=33", "__CAPSTONE_SQLITE_MEMORY_PASSED__"):
            hit = m in out
            ok &= hit
            print(f"  {'OK     ' if hit else 'MISSING'}  {m}")
        print("SQLITE_ON_SILICON:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        try:
            console.power(False); log("powered off")
            if locked:
                console.unlock(); log("unlocked")
        except Exception as e:
            log(f"CLEANUP FAILED ({e}) -- verify the board is off and unlocked")


if __name__ == "__main__":
    sys.exit(main())
