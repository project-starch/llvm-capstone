#!/usr/bin/env python3
"""Read EVERY useful debug register off a wedged core, in one session.

Why this exists. Two days of wedge investigation used exactly two of the ~30 debug-mux
registers (rev-node head/overflow and the dyn-unit status byte) plus gdb register reads --
and the gdb path is demonstrably unreliable here: it returned 0xca11ab1ebadcab1e for $a1 and
0xca00000000 for $mstatus, which is the AXI ERROR-SLAVE response (axi_err_slv.sv:25), i.e.
the read went to an unmapped address and the value was junk. Meanwhile the LED mux carries
the answers on a completely independent path and was never asked.

The decisive one is reg_sel 0b11111 (switches 255):

    debug_led_o = { recent_nontrivial_trap_seen_log_q, recent_nontrivial_mcause_log_q[6:0] }

a LATCHED trap-seen bit. The live $mcause read says 0, but a latched log survives whatever
the machine did afterwards. trap_seen=1 means a trap WAS taken and the monitor's handler
never reported it; trap_seen=0 means no trap ever happened and the core is genuinely stalled.
Those two have opposite fixes and everything so far has been argued rather than measured.

Second most useful: reg_sel 0b00000 (switches 224):

    { ex_commit.valid, load_syncer_req_set, store_syncer_req_set, lsu_ready_ex_id,
      capstone_dyn_ready_ex_id, capstone_flu_ready_ex_id, flush_ctrl_if, priv==M }

capstone_flu_ready_ex_id tests the FLU-dispatch-hang theory directly (a handler that fails to
complete its `send ep.res(...)` never re-asserts rtr, so issue stalls forever with the dyn
unit reporting no waits -- exactly the observed 0x84). priv==M settles the privilege question
that decides whether load_store_unit.sv:933's M-mode-only capability check even applies.

And reg_sel 0b00110..0b01101 give commit_instr_id_commit[0].pc byte by byte -- the COMMITTED
pc, off the LED path rather than through the debug module.

All selectors verified against cva6.sv:1090-1215. debug_byte_sel must be 0b111 (switch bits
7:5 set) for the status bank, so every switch value below is 0b111xxxxx = 224 + reg_sel.
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
SETTLE = float(os.environ.get("PROBE_SETTLE") or 60)
PCC_BASE = int(os.environ.get("PROBE_PCC_BASE") or "0x81E00000", 16)

# (switch value, label, decoder) -- switch = 224 + reg_sel, byte_sel=0b111 throughout.
READS = [
    (255, "TRAP LOG {seen, mcause[6:0]}", "trap"),
    (224, "{excommit,ldsync,stsync,lsu_rdy,dyn_rdy,flu_rdy,flush,privM}", "ready"),
    (225, "{tbe,wstore,wload,wrev,domsw,stall,memwr,memwait}", "status"),
    (249, "rev_node_head[7:0]", "raw"),
    (250, "{overflow,0,head[9:8]}", "raw"),
    (251, "rev_node_serving_idx[7:0]", "raw"),
]
# commit pc, bytes 0..7 -> reg_sel 0b00110..0b01101 == switches 230..237
PC_READS = [(230 + i, i) for i in range(8)]


def log(m):
    print(f"[wedgeregs] {m}", file=sys.stderr, flush=True)


def set_switches(console, value):
    for bit in range(8):
        console.set_switch(bit, bool(value & (1 << bit)))
    time.sleep(1.2)


def read_leds(console):
    st = console.latest(C.LISTEN.get("led_state", "led_state"))
    if isinstance(st, dict):
        bits = st.get("states") or []
        if bits:
            return sum((1 << i) for i, b in enumerate(bits) if b)
    return None


def decode(kind, v):
    if v is None:
        return "UNREAD"
    if kind == "trap":
        seen = (v >> 7) & 1
        cause = v & 0x7f
        return (f"trap_seen={seen} mcause={cause}"
                + ("   <== A TRAP WAS TAKEN; the monitor never reported it"
                   if seen else "   <== NO trap ever taken: a genuine pipeline stall"))
    if kind == "ready":
        names = ["privM", "flush_ctrl_if", "flu_ready", "dyn_ready",
                 "lsu_ready", "store_syncer_req", "load_syncer_req", "ex_commit.valid"]
        return " ".join(f"{n}={(v >> i) & 1}" for i, n in enumerate(names))
    if kind == "status":
        names = ["mem_wait", "mem_write", "stall_issue", "dom_switch_busy",
                 "wait_rev_res", "wait_load_syncer", "wait_store_syncer", "trace_buf_empty"]
        return " ".join(f"{n}={(v >> i) & 1}" for i, n in enumerate(names))
    return ""


def main():
    if not URL:
        raise SystemExit("FPGA_URL not set")
    assert_firmware_embeds_current_initramfs(IMG)
    console = FpgaConsole(URL, logger=lambda m: print(f"[fpga] {m}", file=sys.stderr))
    console.connect()
    install_resilient_emit(console)
    try:
        console.lock()
        install_release_on_signal(console, switches_off=lambda: set_switches(console, 0))
        rb = nvbit(console)
        if rb != BITSTREAM:
            raise SystemExit(f"HARD STOP: resident bitstream is {rb!r}, expected {BITSTREAM!r}")
        console.upload_boot_image(IMG_NAME, str(IMG))
        cold_boot(console, C.GDB_PROMPT, IMG_NAME)
        log("booted; launching the domain (expected to wedge), then reading the mux")

        try:
            console.run_command(f"{HOST} {DOM}; echo D''N_$?", r"DN_\d",
                                timeout=SETTLE, idle_timeout=SETTLE)
            log("NOTE: the run COMPLETED -- reading the registers anyway")
        except Exception as exc:
            log(f"wedged as expected ({type(exc).__name__})")

        print("\n=== WEDGED-CORE REGISTERS (LED mux, independent of the debug module) ===",
              flush=True)
        for sw, label, kind in READS:
            set_switches(console, sw)
            v = read_leds(console)
            shown = "UNREAD" if v is None else f"0x{v:02x} {v:08b}"
            print(f"  sw={sw:3}  {label:52} {shown}", flush=True)
            d = decode(kind, v)
            if d:
                print(f"           {d}", flush=True)

        pc = 0
        got = True
        for sw, idx in PC_READS:
            set_switches(console, sw)
            v = read_leds(console)
            if v is None:
                got = False
                break
            pc |= v << (8 * idx)
        if got:
            va = 0x10000 + (pc - PCC_BASE)
            print(f"\n  COMMIT pc = 0x{pc:016x}  ->  image VA 0x{va:x}", flush=True)
        else:
            print("\n  COMMIT pc: UNREAD", flush=True)
        return 0
    finally:
        print("PROBE_DONE", flush=True)
        release_board(console, switches_off=lambda: set_switches(console, 0),
                      label="wedge regs")


if __name__ == "__main__":
    hard_exit(main())
