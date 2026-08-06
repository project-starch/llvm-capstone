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
import os
import sys, os, pathlib, re, time, hashlib

DRV = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(DRV.parent))
from fpga_driver import config as C
from fpga_driver.fpga_console import FpgaConsole
from fpga_driver.safe_cleanup import release_board, hard_exit, install_release_on_signal
from fpga_driver.run_ladder_perf_fpga import cold_boot, nvbit, sh, install_resilient_emit

URL = os.environ.get("FPGA_URL")
IMG = pathlib.Path(os.environ["FPGA_FW"])
def _hash_name(path: pathlib.Path) -> str:
    """Name the stored boot image by CONTENT, not by a timestamp.

    Callers were passing FPGA_FW_NAME="fw_xx$(date +%H%M%S).bin" to dodge stale entries in
    the board's image store. That works but GUARANTEES a fresh 17.4 MB HTTPS upload every
    run (~30-90 s), and a timestamp proves nothing about freshness anyway. A content hash
    proves everything: identical firmware reuses the stored image, changed firmware gets a
    new name automatically. 18 uploads today, several byte-identical."""
    import hashlib
    h = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"fw_{h}.bin"


IMG_NAME = os.environ.get("FPGA_FW_NAME") or _hash_name(IMG)


def assert_firmware_embeds_current_initramfs(fw: pathlib.Path, locals_=None) -> None:
    """Refuse to flash a firmware that carries a STALE initramfs.

    Measured 2026-07-30, and it silently invalidated a run: `make build A=opensbi-rebuild`
    LINKS fw_payload.bin and only THEN lets buildroot regenerate build/images/, so the
    firmware embeds the PREVIOUS generation kernel+initramfs. The freshly staged
    sqlite_host.user was present in rootfs.cpio and absent from the firmware, and the board
    answered `-sh: /test-domains/sqlite_host.user: not found` (exit 127) -- i.e. the run
    tested nothing while looking like a domain failure. Relinking a second time, with the
    images already final, fixes it.

    Because of that ordering, RE-RUNNING the same target does not converge: each invocation
    relinks and then regenerates the images again. What works is relinking ONCE MORE after a
    generation whose images already contain what you staged -- so this check has to compare
    against the artifact under test, not against a timestamp.

    Three tempting checks that do NOT work, all tried:
      * mtimes -- fw/gz/Image land ~13 s apart in ONE `make`, which reads as "same build";
      * `fw.find(Image[:4096])` -- those bytes are the kernel header, identical across
        generations, so it returns True for a stale firmware;
      * `fw.find(rootfs.cpio.gz[:4096])` -- GZIP EMBEDS AN MTIME, so a freshly regenerated
        .gz is byte-different from the one inside the firmware even when the packed files
        are identical. This reports every firmware stale, including correct ones.

    So: locate the gzip member inside the firmware, DECOMPRESS it, and look for the actual
    binaries we are about to run. That is invariant to compression metadata and to which
    build generation produced it.
    """
    import zlib
    blob = fw.read_bytes()
    off = 0
    while True:
        i = blob.find(b"\x1f\x8b\x08", off)
        if i < 0:
            log(f"WARNING: no initramfs found in {fw.name}, freshness NOT verified")
            return
        off = i + 1
        try:
            cpio = zlib.decompressobj(31).decompress(blob[i:])
        except zlib.error:
            continue
        if len(cpio) > 100_000:      # skip small incidental gzip-looking members
            break

    # WHICH local files to verify. Defaulting to the canonical sqlite_silicon.dom is wrong for
    # any run whose domains are named something else: on 2026-08-06 a staged split (qr18/qr19/
    # qr16n) was blocked as "STALE FIRMWARE" because a LEFTOVER sqlite_silicon.dom from an
    # older build sat at the default path and was, correctly, not in the initramfs. The gate
    # was right about its question and wrong about its subject. Callers that know what they
    # are running pass it in; the default is unchanged for callers that do not.
    missing = []
    for local in (locals_ if locals_ is not None else (LOCAL_HOST, LOCAL_DOM)):
        if not local.is_file():
            continue
        if cpio.find(local.read_bytes()) < 0:
            missing.append(local.name)
    if missing:
        raise SystemExit(
            f"STALE FIRMWARE -- {fw.name} embeds an initramfs whose {', '.join(missing)} "
            f"differ(s) from the local build.\nRe-link it now that the images are final:\n"
            f"  make build LINUX_PAYLOAD=1 A=opensbi-rebuild CAPSTONE_CC_PATH=...\n"
            f"Flashing this runs the PREVIOUS generation and proves nothing (it answers "
            f"`-sh: ...: not found`, exit 127, which looks like a domain failure).")
    log(f"firmware carries the current binaries (initramfs {len(cpio)} bytes, verified "
        f"by decompressed content)")


# Resident-bitstream guard. Reflashed 2026-08-04 to caplifive_fixed_forward.bit, which
# carries the operand-forwarding fix (capstone-ariane 7aac52f93). Overridable so the next
# reflash needs no code change mid-session -- the guard exists to stop a run from silently
# measuring the WRONG silicon, and on 2026-08-04 it did exactly that.
BITSTREAM = os.environ.get("FPGA_BITSTREAM", "caplifive_fixed_forward.bit")
TMP = pathlib.Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone"))
LOCAL_DOM = TMP / "sqlite-silicon" / "sqlite_silicon.dom"
LOCAL_HOST = TMP / "sqlite-build" / "sqlite_host.user"

# Overridable so a probe can be run against the ALREADY-BAKED image without a firmware
# rebuild (which costs ~35 min: stage -> linux-rebuild-with-initramfs ->
# modcapstone-rebuild -> opensbi relink). The decisive use is pointing SQLITE_DOM at a
# nonexistent file so create_dom fails early: if the host's failure message still prints,
# fprintf works and the wedge is state left behind by a SUCCESSFUL create_dom; if that
# print also truncates, stdio/glibc is implicated independently of create_dom.
DOM = os.environ.get("SQLITE_DOM") or "/test-domains/sqlite_silicon.dom"
HOST = os.environ.get("SQLITE_HOST") or "/test-domains/sqlite_host.user"
RUN_TIMEOUT = int(os.environ.get("SQLITE_RUN_TIMEOUT") or 300)
# Abort as soon as the UART goes quiet: a wedged domain emits NOTHING, so the whole
# RUN_TIMEOUT is spent waiting for output that will never arrive. Measured 2026-07-30: a
# run sat idle 10.5 minutes of a 15-minute budget with the log frozen, and each such run
# costs a board session. Any progress resets the clock, so a slow-but-live workload is
# unaffected -- only silence trips it. A working run emits its markers within seconds.
#
# CAVEAT ADDED 2026-07-31 -- the premise "silence means wedged" no longer holds and 75 is
# now too aggressive for a real run. It was written when the domain froze at a pinned pc,
# where silence really was terminal. Since the linear-safe string primitives landed the
# core EXECUTES (pc advances under stepi), and SQLite between `SQ: G/enter` and its first
# row is legitimately silent while it opens the database and runs CREATE/INSERT. A run
# aborted at 75 s idle on exactly that stretch and was nearly read as another wedge.
# Distinguish the two with probe_sqlite_wedge.py + PROBE_STEPI=1 -- an advancing pc means
# raise SQLITE_RUN_IDLE, not that the domain is dead. Default kept low so a genuine wedge
# still fails fast; raise it explicitly when the core is known to be live.
RUN_IDLE = int(os.environ.get("SQLITE_RUN_IDLE") or 75)

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
    assert_firmware_embeds_current_initramfs(IMG)
    local_size = LOCAL_DOM.stat().st_size if LOCAL_DOM.is_file() else -1
    log(f"local build: {LOCAL_DOM} = {local_size} bytes")
    if os.environ.get("SQLITE_DOM"):
        log(f"PROBE MODE: running {HOST} {DOM} -- freshness check skipped")

    console = FpgaConsole(URL, logger=lambda m: print(f"[fpga] {m}", file=sys.stderr))
    console.connect()
    install_resilient_emit(console)
    locked = False
    rc = 1
    try:
        console.lock(); locked = True
        # Arm BEFORE any long wait: a killed driver must still release the board.
        install_release_on_signal(console)
        rb = nvbit(console)
        if rb != BITSTREAM:
            raise SystemExit(f"HARD STOP: resident bitstream is {rb!r}, expected "
                             f"{BITSTREAM!r}. Re-flashing a shared board is never "
                             "automatic -- ask first.")
        log(f"took the lock; bitstream = {rb!r}")
        log(f"uploading firmware ({IMG.stat().st_size} bytes)")
        console.upload_boot_image(IMG_NAME, str(IMG))
        # IMG_NAME must be passed: cold_boot() lives in run_ladder_perf_fpga and a bare
        # IMG_NAME there is the LADDER's hardcoded image name, not ours.
        cold_boot(console, C.GDB_PROMPT, IMG_NAME)
        log("booted to a root shell with /dev/capstone present")

        # Prove the RUNNING system is the image we just uploaded. Verifying the firmware
        # FILE is not enough -- the JTAG load can pull a different stored image, and then
        # every symptom is attributed to the domain. Compare on-board hashes to local ones;
        # `ls` sizes are not enough either (a stale and a current domain were both 1623008
        # bytes on 2026-07-30).
        # SQLITE_LOCAL_DOM names the local file an overridden SQLITE_DOM should match, so a
        # multi-variant session (several .dom baked into one firmware) keeps the stale-boot
        # gate instead of silently losing it. Without it an override skips the check, which
        # is how a run tested the UNCLAMPED domain while the log said "limit=900": staging
        # re-runs build-sqlite-silicon.sh unconditionally (stage-sqlite-in-rootfs.sh:38), so
        # a flag passed only to the earlier standalone build was dropped on the rebuild.
        checks = [(LOCAL_HOST, HOST)] if not os.environ.get("SQLITE_HOST") else []
        if os.environ.get("SQLITE_LOCAL_DOM"):
            checks.append((pathlib.Path(os.environ["SQLITE_LOCAL_DOM"]), DOM))
        elif not os.environ.get("SQLITE_DOM"):
            checks.append((LOCAL_DOM, DOM))
        for local, remote in checks:
            if not local.is_file():
                continue
            want = hashlib.sha256(local.read_bytes()).hexdigest()[:16]
            mark = console.uart_mark()
            console.run_command(f"sha256sum {remote} | cut -c1-16", r"[0-9a-f]{16}", timeout=60)
            got = re.findall(r"\b([0-9a-f]{16})\b", console.uart_since(mark))
            if want not in got:
                raise SystemExit(
                    f"STALE BOOT -- on-board {remote} does not match the local build.\n"
                    f"  local  sha256[:16] = {want}\n  on board            = {got or 'none'}\n"
                    f"The board booted a different stored image than the one uploaded "
                    f"({IMG_NAME}). Do NOT interpret any failure past this point as a "
                    f"domain or monitor bug.")
            log(f"on-board {remote} matches the local build ({want})")

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
        # Mark the buffer BEFORE the command so the diagnostic read is provenance-safe,
        # and dump the run-scoped text on failure too -- otherwise a wedge leaves only the
        # 400-byte tail in the exception and the temptation is to grep the accumulated log,
        # which mixes in output from earlier firmware.
        _mark = console.uart_mark()
        _scoped_path = os.environ.get("SQLITE_SCOPED_OUT") or "/tmp/capstone/sqlite-run-scoped.txt"
        try:
            out = console.run_command(f"{HOST} {DOM}; echo D''N_$?", r"DN_\d",
                                      timeout=RUN_TIMEOUT, idle_timeout=RUN_IDLE)
        finally:
            try:
                with open(_scoped_path, "w") as _f:
                    _f.write(console.uart_since(_mark))
                log(f"run-scoped UART -> {_scoped_path} "
                    f"(THIS RUN ONLY; do not grep the accumulated log)")
            except OSError:
                pass
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
        # COMPLETION SENTINEL -- see probe_sqlite_wedge.py for why this exists: without a
        # definite end-of-run marker, telling a finished session from a hung one means
        # polling `ps`, which is racy and matches the poller's own command line. Printed
        # FIRST so it survives a throwing power-off or unlock.
        print("RUN_DONE", flush=True)
        # Time-boxed: power-off and unlock each wait on a board event and can block
        # forever, which strands the flock and every session queued behind it.
        release_board(console, label="sqlite run")


if __name__ == "__main__":
    # hard_exit, not sys.exit: the board is already released by main()'s finally, but
    # socketio's non-daemon thread can keep the interpreter alive afterwards, which
    # reads from outside as a board session still running.
    hard_exit(main())
