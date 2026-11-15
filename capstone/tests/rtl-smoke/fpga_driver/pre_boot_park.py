#!/usr/bin/env python3
"""Pre-boot switch park, run BEFORE a pinned kit driver boots (runtime lane's interface, 2026-11-14).

The SPEC kits pin their driver by SHA and their board-run.sh by SHA256SUMS, so the verified park in
FpgaConsole.power (switch_park.py) cannot reach them without a pin change. This standalone step runs inside the
caller's booking, right before board-run.sh: connect, read switch_state, park only if a bit is set, read back,
disconnect. No power change, no gdb, no boot.

    python3 pre_boot_park.py <board>        # board 1, 2 or 3
    URL from the file named by CAPSTONE_FPGA_URL_FILE (never printed).

Exit codes (only these three):
     0  clean at connect, nothing emitted;
    10  was set and is now parked (confirmed [0]*8);
     1  park failed, no switch_state seen, or any error: the caller STOPs without booting.
One must-fire line in every case, exceptions included:
    [pre-boot-park] board N states_before=[...] states_after=[...] rc=X
PREBOOT_MOCK_URL (tests only) points it at mock_server with the board check off.
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _redact(s):
    return re.sub(r"[A-Za-z0-9_\-]{24,}", "<REDACTED>", re.sub(r"[a-z]+://\S+", "<URL>", str(s)))


def _states(con, C):
    st = con.latest(C.SWITCH_STATE_EVENT)
    s = st.get(C.SWITCH_STATE_KEY) if isinstance(st, dict) else None
    return list(s) if isinstance(s, list) else None


def park(con, C, SP, first_wait=20.0):
    """Return (rc, before, after) for an already connected console."""
    before = _states(con, C)
    if before is None:
        t0 = time.monotonic()
        while before is None and time.monotonic() - t0 < first_wait:
            time.sleep(0.25)
            before = _states(con, C)
    if before is None:
        return 1, None, None
    if not any(before):
        return 0, before, before
    try:
        after = SP.park_switches(con)
    except SP.SwitchParkFailed:
        return 1, before, _states(con, C)
    return (10 if after and not any(after) else 1), before, after


def main(argv=None, console_factory=None):
    argv = sys.argv[1:] if argv is None else argv
    board = argv[0] if argv else "?"
    rc, before, after, note = 1, None, None, ""
    con = None
    try:
        if board not in ("1", "2", "3"):
            raise ValueError(f"board must be 1, 2 or 3, got {board!r}")
        os.environ["FPGA_BOARD"] = board
        os.environ.setdefault("FPGA_LEASE_PURPOSE", f"pre-boot switch park, board {board} (no boot)")
        from fpga_driver import config as C
        from fpga_driver import fpga_console as FC
        from fpga_driver import switch_park as SP
        if console_factory is not None:
            con = console_factory()
        elif os.environ.get("PREBOOT_MOCK_URL"):
            con = FC.FpgaConsole(os.environ["PREBOOT_MOCK_URL"], allow_unverified=True)
        else:
            path = os.environ.get("CAPSTONE_FPGA_URL_FILE")
            if not path:
                raise RuntimeError("CAPSTONE_FPGA_URL_FILE is not set")
            with open(os.path.expanduser(path)) as fh:
                url = fh.read().strip()
            con = FC.FpgaConsole(url)    # refuses a URL that is not this board's (fpga_console's board check)
        if hasattr(con, "connect"):
            con.connect()
        rc, before, after = park(con, C, SP)
    except BaseException as exc:   # noqa: BLE001 -- every outcome must print the line and exit 0/10/1
        rc, note = 1, f" error={type(exc).__name__}: {_redact(exc)[:200]}"
    finally:
        if con is not None and hasattr(con, "close"):
            try:
                con.close()
            except BaseException:  # noqa: BLE001
                pass
        print(f"[pre-boot-park] board {board} states_before={before} states_after={after} rc={rc}{note}",
              flush=True)
    return rc


if __name__ == "__main__":
    code = main()
    sys.exit(code if code in (0, 1, 10) else 1)
