#!/usr/bin/env python3
"""run_sup_bare-r73.py : R-73's exposure read (board lane, 2026-11-25, at the runtime lane's request). Does the board's
`monitor reset halt` reset the core, i.e. assert ndmreset? The core's rst_ni is ndmreset_n (ariane_xilinx.sv:322-327
rstgen(pll_locked & ~ndmreset) -> :784 i_ariane.rst_ni), and the same rst_ni clears csr_regfile's mscratch_q and mepc_q
to 0 (csr_regfile.sv:3392-3418 at 904cd9c6b) and the adapter's page bitmap and TWC (ISSUES R-73, Exposure).
Named run_sup_bare-*.py so board-busy.py sees it as a board user. No image is loaded and the hart is never resumed.

Sequence, after the bare runner's own power-cycle and gdb attach:
  1. the OpenOCD configuration, for the record: version, targets, reset_config, the reset-assert/-deassert events;
  2. `monitor reset halt` (the boot flow's own command), then pc, mscratch, mepc read with `force` (from the target,
     never OpenOCD's cache);
  3. write mscratch = 0x5a5a5a5a12345678 and mepc = 0x80001234, read both back (CONTROL: the write took; if not, VOID);
  4. `monitor reset halt` again, then pc, mscratch and mepc with `force`.
Predicted, pre-registered here before the run: if reset halt asserts ndmreset, step 4 reads mscratch 0 and mepc 0 (and
pc at the boot ROM, 0x10000); if it does not, step 4 reads the step-3 values back. Anything else is UNDETERMINED.
Env: FPGA_URL, FPGA_BITSTREAM (must equal the resident name), R73_OUT. Always powers off and unlocks."""
import os, sys, time, re, pathlib
DRV = pathlib.Path(os.environ["R73_DRIVER_RS"]) / "fpga_driver"
sys.path.insert(0, str(DRV.parent))
from fpga_driver import config as C                                   # noqa: E402
from fpga_driver.fpga_console import FpgaConsole                      # noqa: E402
from fpga_driver.run_rtl_smoke import POWER_ON_SETTLE, POWER_CYCLE_OFF  # noqa: E402

URL = os.environ.get("FPGA_URL") or sys.exit("FPGA_URL not set")
BITSTREAM = os.environ.get("FPGA_BITSTREAM") or sys.exit("FPGA_BITSTREAM not set")
OUT = pathlib.Path(os.environ["R73_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
MS, ME = 0x5a5a5a5a12345678, 0x80001234


def log(m):
    line = f"[r73 {time.strftime('%H:%M:%S')}] {m}"
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
    console = FpgaConsole(URL)
    console.connect()
    locked = False
    try:
        console.lock(); locked = True
        rb = nvbit(console)
        log(f"resident NV bitstream = {rb!r}")
        if rb != BITSTREAM:
            raise SystemExit(f"HARD STOP: resident bitstream is {rb!r}, expected {BITSTREAM!r}")
        log("power-cycling")
        console.power(False); time.sleep(POWER_CYCLE_OFF)
        console.power(True); time.sleep(POWER_ON_SETTLE)
        P = C.GDB_PROMPT
        console.gdb_start()
        g0 = len(console.gdb_text)
        res = {}

        def cmd(c, t=60.0, key=None):
            try:
                o = console.gdb_cmd(c, P, timeout=t)
            except Exception as e:
                o = f"<{type(e).__name__}: {e}>"
            log(f"$ {c}\n{o.rstrip()}")
            if key:
                m = re.findall(r"0x[0-9a-fA-F]+", o.split("\n", 1)[-1])
                res[key] = int(m[-1], 16) if m else None
            return o

        log("== 1. configuration")
        for c in ("monitor version", "monitor targets", "monitor reset_config",
                  "monitor [target current] cget -event reset-assert",
                  "monitor [target current] cget -event reset-assert-pre",
                  "monitor [target current] cget -event reset-deassert-post"):
            cmd(c, 30.0)
        log("== 2. reset halt #1"); cmd("monitor reset halt"); time.sleep(4.0)
        for r in ("pc", "mscratch", "mepc"): cmd(f"monitor reg {r} force", 30.0, f"a_{r}")
        log("== 3. write and read back (control)")
        cmd(f"monitor reg mscratch {MS:#x}", 30.0); cmd(f"monitor reg mepc {ME:#x}", 30.0)
        for r in ("mscratch", "mepc"): cmd(f"monitor reg {r} force", 30.0, f"b_{r}")
        log("== 4. reset halt #2"); cmd("monitor reset halt"); time.sleep(4.0)
        for r in ("pc", "mscratch", "mepc"): cmd(f"monitor reg {r} force", 30.0, f"c_{r}")
        (OUT / "gdb.txt").write_text(console.gdb_text[g0:])
        f = lambda v: "None" if v is None else f"{v:#x}"
        log("READINGS " + " ".join(f"{k}={f(v)}" for k, v in sorted(res.items())))
        if res.get("b_mscratch") != MS or res.get("b_mepc") != ME:
            v = "VOID: the control write did not read back"
        elif res.get("c_mscratch") == 0 and res.get("c_mepc") == 0:
            v = "RESET: reset halt resets the core (ndmreset asserted)"
        elif res.get("c_mscratch") == MS and res.get("c_mepc") == ME:
            v = "NO RESET: reset halt leaves the core's state (ndmreset not asserted)"
        else:
            v = "UNDETERMINED"
        log(f"R73 VERDICT {v}")
        try: console.gdb_stop()
        except Exception: pass
        return 0 if v.startswith(("RESET", "NO RESET")) else 1
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
