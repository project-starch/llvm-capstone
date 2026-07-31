#!/usr/bin/env python3
"""Read the revocation-node allocator state off a WEDGED board. Settles R-12 empirically.

R-12 predicts that the rev-node allocator's head is 10 bits starting at 3
(capstone_rev_node.anvil:160,168), so allocation ~#1022 wraps to id 0 and reuses LIVE ids.
Untrimmed SQLite performs 1059 carves -- one `split`, hence one allocation, per global -- so
it should cross that line. Every `stc` then blocks on a revocation-node query with no
timeout (capstone_dyn_unit.anvil:395-404), which is what a wedge with no trap looks like.

Until now R-12 has been PREDICTED, never observed. The design already publishes the state
that decides it, on the debug-LED mux (cva6.sv:874-877):

    debug_byte_sel = switches[7:5]      debug_reg_sel = switches[4:0]

    reg_sel 0b11001 (switches 249) -> rev_node_head[7:0]
    reg_sel 0b11010 (switches 250) -> {overflow, 5'b0, head[9:8]}
    reg_sel 0b00001 (switches 225) -> {trace_buf_empty, wait_store_syncer,
                                       wait_load_syncer, wait_rev_res, dom_switch_busy,
                                       stall_issue, mem_write_flag, mem_wait_flag}

The third one discriminates WHY the core is stuck: wait_rev_res with mem_wait_flag set is a
blocked node-table access; wait_rev_res without it is the unbounded REVOKE_NODE walk, which
is the shape id reuse produces; wait_store_syncer is a dropped LSU result instead.

Read AFTER the wedge and WITHOUT resetting: the state is only meaningful while the core is
still stuck in it.
"""
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fpga_driver import config as C
from fpga_driver.fpga_console import FpgaConsole
from fpga_driver.run_ladder_perf_fpga import cold_boot, nvbit, install_resilient_emit
from fpga_driver.run_sqlite_baked_fpga import (
    IMG, IMG_NAME, BITSTREAM, assert_firmware_embeds_current_initramfs)

URL = os.environ.get("FPGA_URL")
DOM = os.environ.get("SQLITE_DOM") or "/test-domains/sqlite_silicon.dom"
HOST = os.environ.get("SQLITE_HOST") or "/test-domains/sqlite_host.user"
SETTLE = float(os.environ.get("PROBE_SETTLE") or 60)

# (label, switch value, how to read the 8 LED bits)
SELECTORS = [
    ("rev_node_head[7:0]",              249),
    ("{overflow,0,head[9:8]}",          250),
    ("{tbe,wstore,wload,wrev,domsw,stall,memwr,memwait}", 225),
    ("rev_node_serving_idx[7:0]",       251),
]


def log(m):
    print(f"[revnode] {m}", file=sys.stderr, flush=True)


def set_switches(console, value):
    for bit in range(8):
        console.set_switch(bit, bool(value & (1 << bit)))
    time.sleep(1.5)


def read_leds(console):
    st = console.latest(C.LISTEN.get("led_state", "led_state"))
    if isinstance(st, dict):
        bits = st.get("states") or []
        if bits:
            # states[0] is LED0; assemble LSB-first into a byte.
            v = sum((1 << i) for i, b in enumerate(bits) if b)
            return v, bits
    return None, st


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
            console.run_command(f"{HOST} {DOM}; echo D''N_$?", r"DN_\d",
                                timeout=SETTLE, idle_timeout=SETTLE)
            log("NOTE: the run COMPLETED -- reading the allocator state anyway")
        except Exception as exc:
            log(f"wedged as expected ({type(exc).__name__}); reading the debug mux")
        with open(os.environ.get("PROBE_SCOPED_OUT") or "/tmp/capstone/revnode-scoped.txt",
                  "w") as f:
            f.write(console.uart_since(mark))

        for label, sw in SELECTORS:
            set_switches(console, sw)
            val, raw = read_leds(console)
            if val is None:
                print(f"[read] {label:52} UNREAD ({raw})", flush=True)
            else:
                print(f"[read] {label:52} switches={sw:3} -> 0x{val:02x} "
                      f"({val:3}) bits={raw}", flush=True)
        return 0
    finally:
        # COMPLETION SENTINEL -- see probe_sqlite_wedge.py for why this exists: without a
        # definite end-of-run marker, telling a finished session from a hung one means
        # polling `ps`, which is racy and matches the poller's own command line. Printed
        # FIRST so it survives a throwing power-off or unlock.
        print("PROBE_DONE", flush=True)
        try:
            set_switches(console, 0)
        except Exception:
            pass
        try:
            console.power(False); log("powered off")
        finally:
            if locked:
                console.unlock(); log("unlocked")


if __name__ == "__main__":
    sys.exit(main())
