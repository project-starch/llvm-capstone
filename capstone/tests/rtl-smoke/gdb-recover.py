#!/usr/bin/env python3
"""Clear an orphaned server-side GDB session, and report the board's live state.

WHY THIS LIVES IN THE REPO. HOW-TO-LAUNCH-ON-FPGA.md points at a recovery script under
/tmp/capstone -- which a server reset wipes, exactly when it is most needed. It was missing on
2026-08-14 for that reason. Scratch is not where a recovery tool belongs.

WHAT IT FIXES. `release_board()` does switches -> power off -> unlock -> close and does NOT call
`gdb_stop()` (safe_cleanup.py:75-92). So a driver can print BOARD_RELEASED with the server's
gdb_state still `running`. `gdb_start()` no-ops unless gdb_state is idle|error, so the NEXT run
burns its timeout before load_image. That state SURVIVES A POWER CYCLE and a lock release --
only gdb_stop() clears it (HOW-TO-LAUNCH-ON-FPGA.md:81,527-528).

IT ALSO ANSWERS "is the board actually looping?" -- the console replays its ring buffer (~524 KB,
about ten prior boots) on every connect, so a GUI showing the bootrom banner over and over is
usually REPLAY, not a live loop. The bootrom prints its banner exactly once per reset and never
loops internally, so N banners means N resets -- and replayed resets look identical to live ones.
This waits past the replay and reports how many NEW bytes arrive, which is the cheap
discriminator. On 2026-08-14 a healthy boot was killed for want of it.

Read-only unless --stop is given.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpga_driver.fpga_console import FpgaConsole   # noqa: E402

DRAIN_S = 12.0     # long enough for the connect-time history replay to finish
WATCH_S = 40.0     # then measure only what arrives live


def main() -> int:
    url = os.environ.get("FPGA_URL")
    if not url:
        print("FPGA_URL not set", file=sys.stderr)
        return 2
    do_stop = "--stop" in sys.argv

    con = FpgaConsole(url)
    con.connect()
    try:
        time.sleep(DRAIN_S)
        before = len(con._uart)
        banners_before = con._uart.count("initializing SD")
        time.sleep(WATCH_S)
        new = len(con._uart) - before
        new_banners = con._uart.count("initializing SD") - banners_before

        print(f"replayed: {before} bytes, {banners_before} bootrom banners")
        print(f"live:     {new} bytes, {new_banners} new banners in {WATCH_S:.0f}s")
        if new_banners > 1:
            print("VERDICT: LIVE BOOT LOOP -- the core is resetting repeatedly")
        elif new < 200:
            print("VERDICT: IDLE. The banners are REPLAY. Nothing is wrong with the board.")
        else:
            print("VERDICT: live output, but not a boot loop")

        if do_stop:
            print("issuing gdb_stop() ...")
            try:
                con.gdb_stop()
                print("gdb_stop: ok -- gdb_state should now be idle for the next run")
            except Exception as exc:
                print(f"gdb_stop: FAILED ({type(exc).__name__}: {exc})")
        else:
            print("(read-only; pass --stop to clear an orphaned GDB session)")
    finally:
        try:
            con.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
