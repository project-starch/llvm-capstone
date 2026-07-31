#!/usr/bin/env python3
"""Is the domain PROGRESSING or SPINNING? Sample pc over wall-clock, not back-to-back.

probe_sqlite_wedge.py answers a different question. Its three `stepi`s are microseconds
apart, so they prove only that the core retires instructions -- which since the
linear-safe string primitives landed it demonstrably does (pc advanced 0x14cc74 ->
0x14cc78 -> 0x14cc7c). That is exactly as consistent with SQLite doing useful work as with
strlen spinning forever on a string whose NUL is unreachable, and the two have opposite
fixes.

Separating them needs samples spread over SECONDS, plus the loop counter:

  * pc leaves strlen between samples            -> progressing; the wall is elsewhere.
  * pc stays in the strlen loop AND a0 climbs   -> spinning on one string. a0 is the index
                                                   (`li a0, -1` then `addi a0, a0, 1`), so
                                                   its magnitude is the character count --
                                                   millions means no terminator is being
                                                   found, which is a DATA bug, not a
                                                   capability one.
  * pc stays and a0 does NOT climb              -> genuinely stuck after all; re-open the
                                                   wedge investigation.

Reads registers ONLY while halted, and always `monitor halt`/`monitor resume`, never
`reset` -- a reset destroys the state being measured.

CAVEAT on the register reads: this board's debug path is not fully reliable. A previous
dump returned 0xca11ab1ebadcab1e for $a1 and 0xca00000000 for $mstatus; that constant is
the AXI ERROR-SLAVE response (axi_err_slv.sv:25), i.e. the read went to an unmapped
address and the value is junk. So treat any register whose value carries the 0xca11ab1e /
0xbadcab1e signature as UNREAD rather than as data -- flagged automatically below.

    usage: FPGA_URL=... FPGA_FW=<fw_payload.bin> python3 probe_sqlite_progress.py
"""
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fpga_driver import config as C
from fpga_driver.fpga_console import FpgaConsole
from fpga_driver.safe_cleanup import release_board, hard_exit, install_release_on_signal
from fpga_driver.run_ladder_perf_fpga import cold_boot, nvbit, install_resilient_emit
from fpga_driver.run_sqlite_baked_fpga import (
    IMG, IMG_NAME, BITSTREAM, assert_firmware_embeds_current_initramfs)

URL = os.environ.get("FPGA_URL")
DOM = os.environ.get("SQLITE_DOM") or "/test-domains/sqlite_silicon.dom"
HOST = os.environ.get("SQLITE_HOST") or "/test-domains/sqlite_host.user"
# Short: the point is to get to sampling while the domain is still working, not to wait
# for output. Any wait here is wait the samples could have used.
SETTLE = float(os.environ.get("PROBE_SETTLE") or 20)
SAMPLES = int(os.environ.get("PROBE_SAMPLES") or 5)
GAP = float(os.environ.get("PROBE_GAP") or 20)

# The domain's .text is linked at image VA 0x10000 and loaded at this base.
PCC_BASE = int(os.environ.get("PROBE_PCC_BASE") or "0x81E00000", 16)
# strlen's LOOP BODY, not the whole function -- the question is "is it going round and
# round", so the prologue/epilogue must fall outside. In the linear-safe build the loop is
# 14cc74 addi / 14cc78 cincoffset / 14cc7c lbu / 14cc80 bnez, so [14cc74, 14cc84).
# Re-derive with `llvm-objdump -d --disassemble-symbols=strlen` after any relink: these
# addresses move whenever the domain is rebuilt.
STRLEN_LO = int(os.environ.get("PROBE_STRLEN_LO") or "0x14cc74", 16)
STRLEN_HI = int(os.environ.get("PROBE_STRLEN_HI") or "0x14cc84", 16)

BAD = (0xca11ab1ebadcab1e, 0xbadcab1e)


def log(m):
    print(f"[progress] {m}", file=sys.stderr, flush=True)


def va(pc):
    return 0x10000 + (pc - PCC_BASE)


def read_reg(console, expr):
    """Return an int, or None if the read failed OR came back as AXI error-slave junk."""
    start = len(console.gdb_text)
    console._emit("gdb_input", text=f"p/x {expr}\n")
    try:
        m = console.wait_gdb(r"\$\d+ = 0x[0-9a-fA-F]+", timeout=25.0, search_from=start)
    except Exception:
        return None
    try:
        v = int(m.group(0).split("=")[1].strip(), 16)
    except ValueError:
        return None
    return None if v in BAD else v


def main():
    if not URL:
        raise SystemExit("FPGA_URL not set")
    assert_firmware_embeds_current_initramfs(IMG)

    console = FpgaConsole(URL, logger=lambda m: print(f"[fpga] {m}", file=sys.stderr))
    console.connect()
    install_resilient_emit(console)
    try:
        console.lock()
        install_release_on_signal(console)
        rb = nvbit(console)
        if rb != BITSTREAM:
            raise SystemExit(f"HARD STOP: resident bitstream is {rb!r}, expected {BITSTREAM!r}")
        console.upload_boot_image(IMG_NAME, str(IMG))
        cold_boot(console, C.GDB_PROMPT, IMG_NAME)
        log("booted; launching the domain, then sampling while it runs")

        try:
            console.run_command(f"{HOST} {DOM}; echo D''N_$?", r"DN_\d",
                                timeout=SETTLE, idle_timeout=SETTLE)
            log("NOTE: the run COMPLETED within the settle window -- nothing to sample")
        except Exception:
            pass

        console.gdb_start()
        rows = []
        try:
            for i in range(SAMPLES):
                console.gdb_cmd("monitor halt", C.GDB_PROMPT, timeout=30.0)
                pc = read_reg(console, "$pc")
                a0 = read_reg(console, "$a0")
                ra = read_reg(console, "$ra")
                rows.append((time.time(), pc, a0, ra))
                where = "UNREAD"
                if pc is not None:
                    v = va(pc)
                    where = f"VA {v:#x}" + (" [in strlen]" if STRLEN_LO <= v < STRLEN_HI
                                            else " [OUTSIDE strlen]")
                print(f"[sample {i}] pc={pc if pc is None else hex(pc)} {where}  "
                      f"a0={a0 if a0 is None else a0}  "
                      f"ra={ra if ra is None else hex(ra)}", flush=True)
                if i < SAMPLES - 1:
                    console.gdb_cmd("monitor resume", C.GDB_PROMPT, timeout=20.0)
                    time.sleep(GAP)
        finally:
            console.gdb_stop()

        # Verdict, stated only where the evidence supports one.
        pcs = [r[1] for r in rows if r[1] is not None]
        a0s = [r[2] for r in rows if r[2] is not None]
        print("\n=== VERDICT ===", flush=True)
        if len(pcs) < 2:
            print("INSUFFICIENT: fewer than two pc reads succeeded.", flush=True)
        elif any(not (STRLEN_LO <= va(p) < STRLEN_HI) for p in pcs):
            print("PROGRESSING: at least one sample is OUTSIDE strlen -- the domain is "
                  "moving through code. The wall is not strlen.", flush=True)
        elif len(a0s) >= 2 and a0s[-1] > a0s[0]:
            print(f"SPINNING IN STRLEN: index a0 climbed {a0s[0]} -> {a0s[-1]} across "
                  f"{len(a0s)} samples while pc stayed in the loop. No terminator is being "
                  f"found -- a DATA problem (bounds/contents of the string), not a "
                  f"capability one.", flush=True)
        elif len(a0s) >= 2:
            print(f"STUCK: pc stayed in strlen and a0 did NOT climb ({a0s}). Re-open the "
                  f"wedge investigation.", flush=True)
        else:
            print("pc stayed in strlen; a0 unread, so spinning vs stuck is UNRESOLVED.",
                  flush=True)
        return 0
    finally:
        print("PROBE_DONE", flush=True)
        release_board(console, label="progress probe")


if __name__ == "__main__":
    hard_exit(main())
