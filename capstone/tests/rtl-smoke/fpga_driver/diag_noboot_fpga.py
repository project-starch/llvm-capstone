#!/usr/bin/env python3
"""Diagnose a boot that downloads completely and then prints nothing (2026-10-19: four in a row on vp2).

One board session, the cold_boot sequence up to `continue`, plus three readings that separate the explanations:
  1. after load_image, `monitor verify_image` of the same stored image at 0x80000000, and the first words at
     0x80000000 read back and compared with the file -- a mismatch says the image did not land in DDR;
  2. ~DIAG_SETTLE s after `continue`, the UART bytes that arrived (none, in the failure being diagnosed);
  3. then `monitor halt` (never reset: a reset zeroes the trap CSRs) and pc, mcause, mepc, mtval, mstatus, a0 read
     once, and twice more across resume/halt: pc in the bootrom (0x10000..) says the core never left it; pc in the
     trap vector or a non-zero mcause says an early fault; a pc moving inside 0x80000000.. says the firmware runs and
     the console path is what is silent.
Every reading is printed as `[diag] ...`; the session ends with DIAG_DONE and releases the board (power off).
Env: FPGA_URL, FPGA_FW (the firmware file), FPGA_BITSTREAM (the expected resident name), DIAG_SETTLE (default 12).
"""
import hashlib, os, pathlib, struct, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fpga_driver import config as C
from fpga_driver.fpga_console import FpgaConsole
from fpga_driver.safe_cleanup import release_board, install_release_on_signal
from fpga_driver.run_ladder_perf_fpga import nvbit, install_resilient_emit
from fpga_driver.run_rtl_smoke import POWER_CYCLE_OFF, POWER_ON_SETTLE

URL = os.environ.get("FPGA_URL")
IMG = pathlib.Path(os.environ["FPGA_FW"])
BITSTREAM = os.environ["FPGA_BITSTREAM"]
SETTLE = float(os.environ.get("DIAG_SETTLE") or 12)
IMG_NAME = f"fw_{hashlib.sha256(IMG.read_bytes()).hexdigest()[:12]}.bin"


def say(m):
    print(f"[diag] {m}", flush=True)


def gdb_value(console, expr, timeout=25.0):
    start = len(console.gdb_text)
    console._emit("gdb_input", text=f"p/x {expr}\n")
    try:
        m = console.wait_gdb(r"\$\d+ = 0x[0-9a-fA-F]+", timeout=timeout, search_from=start)
        return m.group(0).split("=")[1].strip()
    except Exception as exc:
        return f"UNREAD ({type(exc).__name__})"


def gdb_text(console, cmd, timeout=60.0):
    start = len(console.gdb_text)
    console._emit("gdb_input", text=cmd + "\n")
    try:
        console.wait_gdb(C.GDB_PROMPT, timeout=timeout, search_from=start)
    except Exception:
        pass
    return console.gdb_text[start:]


def main():
    if not URL:
        raise SystemExit("FPGA_URL not set")
    words = struct.unpack("<8I", IMG.read_bytes()[:32])
    console = FpgaConsole(URL, logger=lambda m: print(f"[fpga] {m}", file=sys.stderr))
    console.connect()
    install_resilient_emit(console)
    try:
        console.lock()
        install_release_on_signal(console)
        rb = nvbit(console)
        say(f"resident bitstream {rb!r} (expected {BITSTREAM!r})")
        if rb != BITSTREAM:
            raise SystemExit("HARD STOP: resident bitstream differs")
        say(f"image {IMG} sha256/12 {IMG_NAME[3:15]} size {IMG.stat().st_size}")
        console.upload_boot_image(IMG_NAME, str(IMG))
        for attempt in range(1, 4):  # as the baked runner's cold_boot: gdb_start can time out after a power cycle
            console.power(False); time.sleep(POWER_CYCLE_OFF)
            console.power(True); time.sleep(POWER_ON_SETTLE)
            try:
                console.gdb_start()
                say(f"gdb started on attempt {attempt}")
                break
            except Exception as exc:
                say(f"gdb_start failed on attempt {attempt} ({type(exc).__name__})")
                if attempt == 3:
                    raise
        try:
            console.gdb_cmd("monitor reset halt", C.GDB_PROMPT, timeout=60.0)
            time.sleep(4.0)
            say(f"pc after reset halt = {gdb_value(console, '$pc')}")
            out = gdb_text(console, f"monitor load_image images/{IMG_NAME} 0x80000000 bin", timeout=300.0)
            say("load_image: " + " ".join(l.strip() for l in out.splitlines() if "bytes" in l)[:200])
            out = gdb_text(console, f"monitor verify_image images/{IMG_NAME} 0x80000000 bin", timeout=300.0)
            say("verify_image: " + " | ".join(l.strip() for l in out.splitlines() if l.strip() and "(gdb)" not in l)[-300:])
            out = gdb_text(console, "x/8wx 0x80000000")
            say("DDR at 0x80000000: " + " ".join(out.split()[-9:]))
            say("file at offset 0:   " + " ".join(f"0x{w:08x}" for w in words))
            console.gdb_cmd("set $pc = 0x80000000", C.GDB_PROMPT)
            console.gdb_cmd("set $a0 = 0", C.GDB_PROMPT)
            mark = console.uart_mark()
            console._emit("gdb_input", text="continue\n")
            time.sleep(SETTLE)
            uart = console.uart_since(mark)
            say(f"UART bytes in the {SETTLE:.0f} s after continue: {len(uart)} {uart[:200]!r}")
            console.gdb_cmd("monitor halt", C.GDB_PROMPT, timeout=30.0)
            for expr in ("$pc", "$mcause", "$mepc", "$mtval", "$mstatus", "$a0", "$ra", "$sp"):
                say(f"{expr} = {gdb_value(console, expr)}")
            for i in range(2):
                console.gdb_cmd("monitor resume", C.GDB_PROMPT, timeout=20.0)
                time.sleep(2.0)
                console.gdb_cmd("monitor halt", C.GDB_PROMPT, timeout=20.0)
                say(f"pc sample {i + 1} = {gdb_value(console, '$pc')}")
            out = gdb_text(console, "x/6i $pc-8")
            say("around pc: " + " | ".join(l.strip() for l in out.splitlines() if "0x" in l)[:400])
        finally:
            console.gdb_stop()
        return 0
    finally:
        print("DIAG_DONE", flush=True)
        release_board(console, label="no-boot diagnostic")


if __name__ == "__main__":
    sys.exit(main())
