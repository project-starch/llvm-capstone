#!/usr/bin/env python3
"""Run ONE bare M-mode supervised-CALL directed test on the board (no OpenSBI, no Linux).

sup-resume-2026-10-03 copy: the ladder's runner plus the stages driver's WEDGE READ on a timeout -- the LED-path debug
registers (switch 224 {excommit,ldsync,stsync,lsu_rdy,dyn_rdy,flu_rdy,flush,privM}, 225 {tbe,wstore,wload,wrev,domsw,
stall,memwr,memwait}, 255 the latched trap log, 230..237 the committed pc), read AFTER the run (an odd switch value
steals the console TX, so never during one), then the switches parked at 0. Written to <SUP_OUT>/wedge.txt.

The sequence is run_base_bare_fpga.py's, which is proven on this board: upload the raw image,
power-cycle, `monitor reset halt`, JTAG `load_image` at 0x80000000, `set $pc`, `continue`. The image is
the RTL lane's test (capstone-ariane 36a641e0b) wrapped by inc/board_prologue.S and inc/asm_insn.h:
it prints "SUPTEST BEGIN <id>" in integer mode, "SB1" before entering the test, then the test's
readings as "SR <count>" / "SV <value>" lines and "SUPTEST END".

After the end marker or a timeout, the runner sends ^C to GDB and tries to read pc, mcause, mepc, mtval and
board_rec[]. THAT FALLBACK DOES NOT WORK (2026-10-02, all three runs): OpenOCD reports `Unable to halt.
dmcontrol=0x80000001, dmstatus=0x00030c82` (allrunning), and every read returns E0E. The core did not honour the
halt request while it spun in the capability-mode report loop. The UART report is the only working channel, so a
wedged run returns nothing beyond SUPTEST BEGIN / SB1. Fixing this is open.

Env: FPGA_URL, FPGA_BITSTREAM (must equal the resident name), SUP_IMG (.bin), SUP_REC_ADDR (board_rec,
hex), SUP_OUT (output dir), SUP_TIMEOUT (s, default 90). Always powers off and unlocks.
"""
import os, sys, time, re, pathlib, hashlib
DRV = pathlib.Path(__file__).resolve().parent.parent / "fpga_driver"
sys.path.insert(0, str(DRV.parent))
from fpga_driver import config as C                                   # noqa: E402
from fpga_driver.fpga_console import FpgaConsole                      # noqa: E402
from fpga_driver.run_rtl_smoke import POWER_ON_SETTLE, POWER_CYCLE_OFF  # noqa: E402

URL = os.environ.get("FPGA_URL") or sys.exit("FPGA_URL not set")
BITSTREAM = os.environ.get("FPGA_BITSTREAM") or sys.exit("FPGA_BITSTREAM not set")
IMG = pathlib.Path(os.environ["SUP_IMG"])
REC = int(os.environ["SUP_REC_ADDR"], 16)
OUT = pathlib.Path(os.environ["SUP_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
TIMEOUT = float(os.environ.get("SUP_TIMEOUT", "90"))
IMG_NAME = "sup-" + hashlib.sha256(IMG.read_bytes()).hexdigest()[:12] + ".bin"


def log(m):
    line = f"[sup {time.strftime('%H:%M:%S')}] {m}"
    print(line, file=sys.stderr, flush=True)
    with open(OUT / "run.log", "a") as f:
        f.write(line + "\n")


def nvbit(console, poll=8.0):
    end = time.time() + poll
    while True:
        with console._cond:
            v = (console._state.get("flash_state") or {}).get("nv_bitstream_name")
        if v is not None or time.time() >= end:
            return v
        time.sleep(0.5)


def main():
    log(f"image {IMG} -> store name {IMG_NAME}; board_rec at {REC:#x}")
    console = FpgaConsole(URL)
    console.connect()
    locked = False
    rc = 1
    try:
        console.lock(); locked = True
        rb = nvbit(console)
        log(f"resident NV bitstream = {rb!r}")
        if rb != BITSTREAM:
            raise SystemExit(f"HARD STOP: resident bitstream is {rb!r}, expected {BITSTREAM!r}")
        for attempt in range(1, 4):
            try:
                console.upload_boot_image(IMG_NAME, str(IMG)); break
            except Exception as e:
                log(f"upload failed (attempt {attempt}): {e}")
                if not getattr(console.sio, "connected", False):
                    try: console.connect(); time.sleep(1.0)
                    except Exception as e2: log(f"reconnect failed: {e2}")
                time.sleep(3.0)
        else:
            raise SystemExit("upload_boot_image failed on every attempt")
        log("upload complete; power-cycling")
        console.power(False); time.sleep(POWER_CYCLE_OFF)
        console.power(True); time.sleep(POWER_ON_SETTLE)
        prompt = C.GDB_PROMPT
        console.gdb_start()
        log("reset halt"); console.gdb_cmd("monitor reset halt", prompt, timeout=60.0)
        time.sleep(4.0)
        log("load_image at 0x80000000")
        console.gdb_cmd(f"monitor load_image images/{IMG_NAME} 0x80000000 bin", prompt, timeout=300.0)
        console.gdb_cmd("set $pc = 0x80000000", prompt)
        start = len(console.uart_text)
        log("continue")
        console._emit("gdb_input", text="continue\n")
        deadline = time.time() + TIMEOUT
        out = ""
        while time.time() < deadline:
            t = console.uart_text
            out = t[start:] if len(t) >= start else t
            if "SUPTEST END" in out:
                break
            time.sleep(1.0)
        ended = "SUPTEST END" in out
        log(f"end marker {'SEEN' if ended else 'NOT seen -- timeout, reading state through GDB'}")
        if not ended:
            wl = []
            def setsw(v):
                for bit in range(8):
                    console.set_switch(bit, bool(v & (1 << bit)))
                time.sleep(1.2)
            def leds():
                st = console.latest(C.LISTEN.get("led_state", "led_state"))
                bits = (st or {}).get("states") or [] if isinstance(st, dict) else []
                return sum((1 << i) for i, b in enumerate(bits) if b) if bits else None
            try:
                # labels verified against cva6.sv at 36a641e0b (bank 111 = 224+reg, bank 110 = 192+reg), MSB first
                for sw, label in ((255, "TRAP LOG {seen,mcause[6:0]}"),
                                  (224, "{excommit,ldsync,stsync,lsu_rdy,dyn_rdy,flu_rdy,flush,privM}"),
                                  (225, "{tbe,wstore,wload,wrev,domsw,stall,memwr,memwait}"),
                                  (226, "{data_valid,data_ack,data_resp_valid,data_resp_ack,reg_valid,reg_ack,reg_resp_valid,reg_resp_ack}"),
                                  (227, "{commit_dsw_valid,dsw_commit_ack,issue_reg_resp_v,csr_reg_resp_v,frontend_reg_resp_v,data_req.write_en,reg_req.is_set,data_req.metadata_en}"),
                                  (228, "{1,dom_switch_idx[6:0]}"),
                                  (229, "{load_state[3:0],0000}"),
                                  (238, "{busy_seen,pc_loaded_seen,last_data_metadata_en,last_reg_is_set,last_reg_id[3:0]}"),
                                  (239, "{0,dom_switch_last_idx_log[6:0]}"),
                                  (240, "{0,dom_switch_last_reg_id_log[6:0]}"),
                                  (192, "{0000000,commit_instr[0].valid}"),
                                  (193, "{00000,store_buf_commit_cnt}"),
                                  (194, "{000000,store_state}"),
                                  (195, "{0000,load_state}")):
                    setsw(sw); v = leds()
                    wl.append(f"sw={sw} {label} " + ("UNREAD" if v is None else f"0x{v:02x} {v:08b}"))
                pc = 0; ok = True
                for i in range(8):
                    setsw(230 + i); v = leds()
                    if v is None: ok = False; break
                    pc |= v << (8 * i)
                wl.append("commit pc " + (f"0x{pc:016x}" if ok else "UNREAD"))
            except Exception as e:
                wl.append(f"wedge read failed: {e}")
            finally:
                try: setsw(0)
                except Exception: pass
            for l in wl: log(f"[wedge] {l}")
            (OUT / "wedge.txt").write_text("\n".join(wl) + "\n")
        # GDB readout in both cases: interrupt, then registers and the readings buffer in memory.
        g0 = len(console.gdb_text)
        try:
            console._emit("gdb_input", text="\x03")
            console.wait_gdb(prompt, timeout=20.0, search_from=g0)
            for cmd in ("p/x $pc", "p/x $mcause", "p/x $mepc", "p/x $mtval", "p/x $mstatus",
                        f"x/1gx {REC:#x}", f"x/48gx {REC + 8:#x}"):
                console.gdb_cmd(cmd, prompt, timeout=30.0)
        except Exception as e:
            log(f"GDB readout failed: {e}")
        (OUT / "gdb.txt").write_text(console.gdb_text[g0:])
        (OUT / "uart.txt").write_text(out)
        sr = re.findall(r"SR ([0-9A-F]{16})", out)
        sv = re.findall(r"SV ([0-9A-F]{16})", out)
        log(f"UART: begin={'SUPTEST BEGIN' in out} SB1={'SB1' in out} SR={sr} SV lines={len(sv)}")
        for i, v in enumerate(sv, 1):
            log(f"  reading {i:3d}: 0x{v}")
        rc = 0 if ended else 2
        try: console.gdb_stop()
        except Exception: pass
        return rc
    finally:
        if not getattr(console.sio, "connected", False):
            try: console.connect(); time.sleep(1.0); log("reconnected for cleanup")
            except Exception as e: log(f"cleanup reconnect FAILED: {e} -- board may still be locked")
        for attempt in range(3):
            try: console.power(False); log("powered off"); break
            except Exception as e: log(f"power off err (try {attempt+1}): {e}"); time.sleep(2.0)
        if locked:
            for attempt in range(3):
                try: console.unlock(); log("unlocked"); break
                except Exception as e: log(f"unlock err (try {attempt+1}): {e}"); time.sleep(2.0)
        try: console.close()
        except Exception: pass


if __name__ == "__main__":
    sys.exit(main())
