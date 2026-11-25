#!/usr/bin/env python3
"""run_sup_bare-linux.py : the console capture for the first Linux boot on B5b-2 (runtime/b5b2-linux c3a45f1e9904,
capstone/tests/rtl-smoke/b5b2-linux-2026-11-25/PREREG.md). Named run_sup_bare-*.py so board-busy.py sees a board user.
Sequence (cold_boot's, run_ladder_perf_fpga.py:255-264, without the login): resident name check, upload under a
content-hashed name, power-cycle, `monitor reset halt` (R-73's covered case, dev 1e4c200c4c9b), `load_image` at
0x80000000, `set $pc = 0x80000000`, `set $a0 = 0`, `continue`. The capture is scoped to this run's own `continue`
(the console replays the previous boot on connect). Stop: "TRAPTEST: done" then a 20 s drain for the poweroff panic,
or LX_BOUND s (default 900) after the load. Env: FPGA_URL, FPGA_BITSTREAM, LX_IMG, LX_SHA, LX_OUT, LX_DRIVER_RS.
Exit 0 when "TRAPTEST: done" was seen, 2 at the bound, 1 on a stop before the load. Always powers off and unlocks."""
import os, sys, time, pathlib, hashlib
sys.path.insert(0, os.environ["LX_DRIVER_RS"])
from fpga_driver import config as C                                   # noqa: E402
from fpga_driver.fpga_console import FpgaConsole                      # noqa: E402
from fpga_driver.run_rtl_smoke import POWER_ON_SETTLE, POWER_CYCLE_OFF  # noqa: E402

URL = os.environ.get("FPGA_URL") or sys.exit("FPGA_URL not set")
BITSTREAM = os.environ.get("FPGA_BITSTREAM") or sys.exit("FPGA_BITSTREAM not set")
IMG = pathlib.Path(os.environ["LX_IMG"]); SHA = os.environ["LX_SHA"]
OUT = pathlib.Path(os.environ["LX_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
BOUND = float(os.environ.get("LX_BOUND", "900"))


def log(m):
    line = f"[lx {time.strftime('%H:%M:%S')}] {m}"
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
    data = IMG.read_bytes(); h = hashlib.sha256(data).hexdigest()
    if h != SHA:
        log(f"HARD STOP: {IMG.name} sha256 {h[:16]} is not {SHA[:16]}"); return 1
    name = "lx-" + h[:12] + ".bin"
    log(f"image {IMG.name} sha256 {h[:16]} ({len(data)} bytes) -> store name {name}")
    console = FpgaConsole(URL)
    console.connect()
    locked = False; rc = 1; start = None
    try:
        console.lock(); locked = True
        rb = nvbit(console)
        log(f"resident NV bitstream = {rb!r}")
        if rb != BITSTREAM:
            log(f"HARD STOP: resident is not {BITSTREAM!r}"); return 1
        for attempt in range(1, 4):
            try:
                t0 = time.time(); console.upload_boot_image(name, str(IMG)); log(f"upload complete ({time.time() - t0:.0f} s)"); break
            except Exception as e:
                log(f"upload failed (attempt {attempt}): {e}")
                if not getattr(console.sio, "connected", False):
                    try: console.connect(); time.sleep(1.0)
                    except Exception as e2: log(f"reconnect failed: {e2}")
                time.sleep(3.0)
        else:
            log("HARD STOP: upload failed on every attempt"); return 1
        log("power-cycling")
        console.power(False); time.sleep(POWER_CYCLE_OFF)
        console.power(True); time.sleep(POWER_ON_SETTLE)
        P = C.GDB_PROMPT
        console.gdb_start()
        g0 = len(console.gdb_text)
        log("reset halt"); console.gdb_cmd("monitor reset halt", P, timeout=60.0); time.sleep(4.0)
        log("load_image at 0x80000000"); t0 = time.time()
        console.gdb_cmd(f"monitor load_image images/{name} 0x80000000 bin", P, timeout=600.0)
        log(f"load complete ({time.time() - t0:.0f} s)")
        console.gdb_cmd("set $pc = 0x80000000", P); console.gdb_cmd("set $a0 = 0", P)
        start = len(console.uart_text)
        log("continue"); console._emit("gdb_input", text="continue\n")
        t_load = time.time(); deadline = t_load + BOUND; last = 0.0; out = ""; done_at = None
        while time.time() < deadline:
            t = console.uart_text; out = t[start:] if len(t) >= start else t
            if done_at is None and "TRAPTEST: done" in out:
                done_at = time.time(); log("TRAPTEST: done seen; draining 20 s")
            if done_at is not None and time.time() - done_at >= 20.0:
                break
            if time.time() - last > 60:
                log(f"  ... {len(out)} UART chars, {out.count('TRAPTEST:')} TRAPTEST lines"); last = time.time()
            time.sleep(1.0)
        (OUT / "uart.txt").write_text(out)
        (OUT / "gdb.txt").write_text(console.gdb_text[g0:])
        rc = 0 if done_at is not None else 2
        log(f"end: {'TRAPTEST: done seen' if rc == 0 else f'BOUND {BOUND:.0f} s reached without TRAPTEST: done'}; {len(out)} chars")
        try: console.gdb_stop()
        except Exception: pass
        return rc
    finally:
        if start is None:
            try: (OUT / "uart.txt").exists() or log("no capture (stopped before the continue)")
            except Exception: pass
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
