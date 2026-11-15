#!/usr/bin/env python3
"""End-to-end check of the power-on park through the real FpgaConsole and the mock server (no board).

Positive arm: the mock's switches are set to 217 (sw[0]=1, the 2026-11-14 board-2 state), then power(True) must emit
switch_reset_all, see the switches at 0, and only then power on. Negative arm: a clean board powers on with no
switch emit at all. Uses a private BOARD_LOCK so a live board session's lock is never touched. Exit 0 when both hold."""
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ["BOARD_LOCK"] = os.path.join(tempfile.mkdtemp(), "park-mock.board.lock")
from fpga_driver import config as C  # noqa: E402
from fpga_driver.fpga_console import FpgaConsole  # noqa: E402
from fpga_driver.test_dryrun import _MockServer  # noqa: E402


def arm(set_bits):
    lines = []
    with _MockServer() as srv:
        con = FpgaConsole(srv.url, allow_unverified=True, logger=lines.append)
        con.connect(timeout=10)
        try:
            for b in set_bits:
                con.set_switch(b, True)
            before = (con.latest(C.SWITCH_STATE_EVENT) or {}).get(C.SWITCH_STATE_KEY)
            mark = len(lines)
            con.power(True)
            after = (con.latest(C.SWITCH_STATE_EVENT) or {}).get(C.SWITCH_STATE_KEY)
            power = (con.latest(C.POWER_STATE_EVENT) or {}).get("state")
            emits = [l for l in lines[mark:] if l.startswith("emit ")]
        finally:
            con.close()
    return before, after, power, emits


def main():
    # one console per process: the local board lock is released only at process exit (fpga_console, by design)
    if len(sys.argv) > 1:
        which = sys.argv[1]
        before, after, power, emits = arm([0, 3, 4, 6, 7] if which == "217" else [])
        names = [e.split()[1] for e in emits]
        if which == "217":
            ok = (before == [1, 0, 0, 1, 1, 0, 1, 1] and after == [0] * 8 and power == "on"
                  and "switch_reset_all" in names and "power_toggle" in names
                  and names.index("switch_reset_all") < names.index("power_toggle"))
            print(("PASS" if ok else "FAIL") + f" 217 parked before power-on: before={before} after={after} "
                  f"power={power} emits={names}")
        else:
            ok = before == [0] * 8 and power == "on" and not any(n.startswith("switch") for n in names)
            print(("PASS" if ok else "FAIL") + f" clean board: no switch emit: emits={names}")
        sys.exit(0 if ok else 1)
    import subprocess
    rc = [subprocess.run([sys.executable, __file__, w]).returncode for w in ("217", "clean")]
    sys.exit(1 if any(rc) else 0)


main()
