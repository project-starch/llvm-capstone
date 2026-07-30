#!/usr/bin/env python3
"""Halt the WEDGED SQLite domain and read the trap CSRs, in ONE board session.

Why this exists. The share entry wedges at sbi_capstone.c:984
(`__domcallsaves(d, CAPSTONE_DPI_REGION_SHARE, r)`): the monitor calls into the domain and
never gets control back. Crucially NO monitor trap tag (EXCX/ILLX/IRQX) is ever printed, so
either the domain is spinning without faulting, or a domain trap wedges M-mode before the
handler can report -- the documented behaviour of this captype-fixed CVA6. Those two have
opposite fixes, and the UART cannot tell them apart: the domain has no console.

$pc/$mcause/$mepc/$mtval separate them in one read:
  * mcause = 0 and pc inside the domain's .text  -> the domain is LOOPING, not faulting;
    pc names the loop.
  * mcause != 0 with mepc inside the domain      -> the domain TRAPPED at mepc, and the
    monitor's handler never reported it.

run_sqlite_baked_fpga.py cannot be used for this: it powers the board off in its `finally`,
which destroys the very state being probed. So this drives the same boot itself and attaches
gdb while the wedge is still live. `monitor halt`, never `reset` -- a reset zeroes mcause.
"""
import sys, os, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fpga_driver import config as C
from fpga_driver.fpga_console import FpgaConsole
from fpga_driver.run_ladder_perf_fpga import cold_boot, nvbit, install_resilient_emit
from fpga_driver.run_sqlite_baked_fpga import (
    IMG, IMG_NAME, BITSTREAM, assert_firmware_embeds_current_initramfs)

URL = os.environ.get("FPGA_URL")
DOM = os.environ.get("SQLITE_DOM") or "/test-domains/sqlite_silicon.dom"
HOST = os.environ.get("SQLITE_HOST") or "/test-domains/sqlite_host.user"
SETTLE = float(os.environ.get("PROBE_SETTLE") or 45)
OUT = os.environ.get("PROBE_SCOPED_OUT") or "/tmp/capstone/probe-wedge.txt"


def log(m):
    print(f"[wedge] {m}", file=sys.stderr, flush=True)


def main():
    if not URL:
        raise SystemExit("FPGA_URL not set")
    assert_firmware_embeds_current_initramfs(IMG)

    console = FpgaConsole(URL, logger=lambda m: print(f"[fpga] {m}", file=sys.stderr))
    console.connect()
    install_resilient_emit(console)
    locked = False
    try:
        console.lock(); locked = True
        rb = nvbit(console)
        if rb != BITSTREAM:
            raise SystemExit(f"HARD STOP: resident bitstream is {rb!r}, expected {BITSTREAM!r}")
        console.upload_boot_image(IMG_NAME, str(IMG))
        cold_boot(console, C.GDB_PROMPT, IMG_NAME)
        log("booted; launching the domain (expected to wedge)")

        mark = console.uart_mark()
        try:
            # Expected to time out -- that IS the wedge. Keep the window short: the point
            # is to attach while the core is still in whatever state it died in.
            console.run_command(f"{HOST} {DOM}; echo D''N_$?", r"DN_\d",
                                timeout=SETTLE, idle_timeout=SETTLE)
            log("NOTE: the run COMPLETED -- there is no wedge to probe")
        except Exception as exc:
            log(f"wedged as expected ({type(exc).__name__}); attaching gdb")

        with open(OUT, "w") as f:
            f.write(console.uart_since(mark))
        log(f"run-scoped UART -> {OUT}")

        console.gdb_start()
        try:
            console.gdb_cmd("monitor halt", C.GDB_PROMPT, timeout=30.0)
            # Wait for the VALUE, not for the prompt. Waiting on C.GDB_PROMPT returns just
            # the command echo: the previous command's trailing "(gdb) " lands in the buffer
            # after `start` is taken, so the first prompt match is the stale one. gdb prints
            # values as `$<n> = 0x<hex>`, which is unambiguous.
            # Sample pc across resume/halt cycles. mcause/mepc/mtval all read 0 on the first
            # probe, i.e. NO trap was ever taken -- the domain is executing, not faulting.
            # Two pc values that differ prove it is live (a loop) rather than stopped; the
            # disassembly then names the loop. `monitor halt`, never reset: reset zeroes the
            # CSRs that make this readable.
            for i in range(3):
                start = len(console.gdb_text)
                console._emit("gdb_input", text="p/x $pc\n")
                try:
                    m = console.wait_gdb(r"\$\d+ = 0x[0-9a-fA-F]+", timeout=25.0,
                                         search_from=start)
                    print(f"[probe] pc sample {i} = {m.group(0).split('=')[1].strip()}",
                          flush=True)
                except Exception as exc:
                    print(f"[probe] pc sample {i} UNREAD ({type(exc).__name__})", flush=True)
                if i < 2:
                    console.gdb_cmd("monitor resume", C.GDB_PROMPT, timeout=20.0)
                    time.sleep(2.0)
                    console.gdb_cmd("monitor halt", C.GDB_PROMPT, timeout=20.0)

            start = len(console.gdb_text)
            console._emit("gdb_input", text="x/24i $pc-32\n")
            try:
                console.wait_gdb(C.GDB_PROMPT, timeout=30.0, search_from=start)
            except Exception:
                pass
            print("[probe] --- disassembly around pc ---", flush=True)
            print(console.gdb_text[start:], flush=True)

            for expr in ("$ra", "$sp", "$gp", "$a0", "$a1", "$mcause", "$mstatus"):
                start = len(console.gdb_text)
                console._emit("gdb_input", text=f"p/x {expr}\n")
                try:
                    m = console.wait_gdb(r"\$\d+ = 0x[0-9a-fA-F]+", timeout=25.0,
                                         search_from=start)
                    print(f"[probe] {expr} = {m.group(0).split('=')[1].strip()}", flush=True)
                except Exception as exc:
                    print(f"[probe] {expr} UNREAD ({type(exc).__name__})", flush=True)
        finally:
            console.gdb_stop()
        return 0
    finally:
        try:
            console.power(False); log("powered off")
        finally:
            if locked:
                console.unlock(); log("unlocked")


if __name__ == "__main__":
    sys.exit(main())
