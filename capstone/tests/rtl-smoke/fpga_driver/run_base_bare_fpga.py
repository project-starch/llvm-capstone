#!/usr/bin/env python3
"""Run the BARE-METAL baseline on the Capstone CVA6 FPGA (issue I-2).

Boots fw_payload_base_bare.bin -- OpenSBI + FDT with our S-mode program in place of
the Linux Image -- and captures the BASE RESULT lines straight off the UART. There
is no OS, no login, and no file transfer: the measurement IS the boot.

Why this exists: the Linux-hosted baseline services timer interrupts inside the
measurement bracket, which inflates the denominator of every overhead ratio and so
UNDERSTATES capability overhead. Repeating and taking the least-disturbed pass fixes
kernels shorter than a timer tick and does nothing for long ones. This removes the
confound at the source instead of modelling it.

Board etiquette (non-negotiable): verifies the resident bitstream before measuring,
ALWAYS powers off + unlocks in finally. The board URL/token is read from FPGA_URL
and never echoed.
"""
import os
import sys, os, time, re, pathlib

DRV = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(DRV.parent))
from fpga_driver import config as C
from fpga_driver.fpga_console import FpgaConsole, ActionTimeout
from fpga_driver.run_rtl_smoke import POWER_ON_SETTLE, POWER_CYCLE_OFF
from socketio.exceptions import BadNamespaceError


def _board_url():
    u = os.environ.get("FPGA_URL")
    if u:
        return u.strip()
    raise SystemExit("no board URL: export FPGA_URL for this run (do not commit it)")


URL = _board_url()
ART = pathlib.Path(os.environ.get("BASE_BARE_DIR") or
                   os.path.join(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone"),
                                "ladder-base-bare"))
IMG = ART / "fw_payload_base_bare.bin"
IMG_NAME = "fw_payload_base_bare.bin"
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
CAPTURE = "/tmp/capstone/board-run-base-bare.uart.txt"


def log(m): print(f"[run] {m}", file=sys.stderr, flush=True)


def _window(console, start):
    """UART text since `start`, tolerating a buffer reset.

    The console clears its UART buffer when the board reboots, so an index taken
    before `continue` can exceed the new length -- every slice then returns '' and
    a perfectly good run reports "0 UART chars". Fall back to the whole buffer."""
    t = console.uart_text
    return t[start:] if len(t) >= start else t


def nvbit(console, poll=8.0):
    end = time.time() + poll
    while True:
        with console._cond:
            fs = console._state.get("flash_state") or {}
        v = fs.get("nv_bitstream_name")
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
        log(f"took the lock; resident NV bitstream = {rb!r}")
        if rb != BITSTREAM:
            # Same HARD STOP as the other runners: flashing reprograms a SHARED
            # board and needs an explicit human go every time.
            if os.environ.get("FPGA_ALLOW_FLASH") != "1":
                raise SystemExit(
                    f"HARD STOP: resident bitstream is {rb!r}, expected {BITSTREAM!r}.\n"
                    "Ask the user, then re-run with FPGA_ALLOW_FLASH=1 if approved.")

        if not IMG.is_file():
            raise SystemExit(f"missing {IMG} (run build-ladder-base-bare.sh)")
        log(f"uploading {IMG_NAME} ({IMG.stat().st_size} bytes -- the Linux payload "
            f"was ~15 MB, so the JTAG reload should be much faster)")
        # The upload has no internal timeout and hung indefinitely once. Retry on a
        # dead socket rather than stalling forever with no output, which is
        # indistinguishable from progress to anyone watching.
        for attempt in range(1, 4):
            try:
                console.upload_boot_image(IMG_NAME, str(IMG))
                break
            except Exception as e:
                log(f"upload failed (attempt {attempt}): {e}")
                if not getattr(console.sio, "connected", False):
                    try: console.connect(); time.sleep(1.0)
                    except Exception as e2: log(f"reconnect failed: {e2}")
                time.sleep(3.0)
        else:
            raise SystemExit("upload_boot_image failed on every attempt")
        log("upload complete")

        # Progress logging through the previously silent phase. Between "upload
        # complete" and the results there used to be no output at all, which is
        # the exact window the first attempt died in -- and a stalled run then
        # looks identical to a working one.
        log("power-cycling"); console.power(False); time.sleep(POWER_CYCLE_OFF)
        console.power(True); time.sleep(POWER_ON_SETTLE)
        prompt = C.GDB_PROMPT
        log("attaching gdb"); console.gdb_start()
        try:
            log("reset halt"); console.gdb_cmd("monitor reset halt", prompt, timeout=60.0)
            time.sleep(4.0)
            log("JTAG load_image (2 MB, should be far faster than the 15 MB Linux image)")
            console.gdb_cmd(f"monitor load_image images/{IMG_NAME} 0x80000000 bin",
                            prompt, timeout=300.0)
            log("load complete; releasing the hart")
            console.gdb_cmd("set $pc = 0x80000000", prompt)
            console.gdb_cmd("set $a0 = 0", prompt)
            start = len(console.uart_text)
            console._emit("gdb_input", text="continue\n")
        finally:
            pass

        # No login, no shell: the program prints and shuts down. Wait for its own
        # end marker rather than a prompt.
        log("waiting for BARE BASELINE DONE")
        deadline = time.time() + 420
        out = ""
        last_note = 0.0
        while time.time() < deadline:
            if time.time() - last_note > 60:
                seen = _window(console, start)
                log(f"  ... {len(seen)} UART chars so far, "
                    f"{seen.count('BASE RESULT')} result rows")
                last_note = time.time()
            out = _window(console, start)
            if "BARE BASELINE DONE" in out:
                break
            time.sleep(2.0)
        try:
            console.gdb_stop()
        except Exception:
            pass

        if "BARE BASELINE START" not in out:
            log("NO START MARKER -- the payload did not run. Firmware layout or "
                "entry address is wrong; check the payload offset and link address.")
        rows = re.findall(r"BASE RESULT (\S+) pass=(\d+) retval=(\d+) "
                          r"cycles=(\d+) instret=(\d+)", out)
        log(f"parsed {len(rows)} result rows")

        best = {}
        for name, p, rv, cyc, ins in rows:
            best.setdefault(name, []).append((int(p), rv, int(cyc), int(ins)))
        for name, passes in best.items():
            warm = [t for t in passes if t[0] >= 2] or passes
            bi = min(t[3] for t in warm)
            tied = [t for t in warm if t[3] == bi]
            bc = min(t[2] for t in tied)
            spread = max(t[2] for t in warm) - min(t[2] for t in warm)
            log(f"  {name}: BEST cycles={bc} instret={bi} "
                f"({len(tied)}/{len(warm)} passes at min instret, spread={spread}) "
                f"retval={passes[0][1]}")
        pathlib.Path(CAPTURE).write_text(console.uart_text)
        log(f"UART capture -> {CAPTURE}")
        return 0 if rows else 1
    finally:
        # RECONNECT BEFORE CLEANUP. Without this the safety net does not work: on
        # 2026-07-28 this runner hung in upload_boot_image, and when interrupted
        # both power(False) and unlock() failed with "not a connected namespace"
        # because the socket had already died -- leaving the board LOCKED and
        # POWERED ON for the next person. The other runners already do this; this
        # one did not, and the omission is exactly what a finally block is for.
        if not getattr(console.sio, "connected", False):
            try:
                console.connect(); time.sleep(1.0); log("reconnected for cleanup")
            except Exception as e:
                log(f"cleanup reconnect FAILED: {e} -- board may still be locked")
        for attempt in range(3):
            try: console.power(False); log("powered off"); break
            except Exception as e: log(f"power off err (try {attempt+1}): {e}")
            time.sleep(2.0)
        if locked:
            for attempt in range(3):
                try: console.unlock(); log("unlocked"); break
                except Exception as e: log(f"unlock err (try {attempt+1}): {e}")
                time.sleep(2.0)
        try: console.close()
        except Exception: pass


if __name__ == "__main__":
    sys.exit(main())
