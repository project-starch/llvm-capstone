#!/usr/bin/env python3
"""Run SEVERAL staged SQLite domains in ONE boot, and report where the sequence breaks.

Why this exists. Six board sessions were spent narrowing a wedge inside strlen, and each
one bought a single bit: a wedge produces no output at all, so the only thing a failed run
says is "somewhere after SQ: G/enter". The clamp experiment eventually showed strlen was
not even spinning, i.e. all six sessions had been bisecting the wrong thing.

Staged-return builds fix the information problem: each one runs the first N steps of
run_sqlite() and RETURNS a marker (0x5A6E_ssrr, ss = stage, rr = the SQLite rc), which the
host prints as `SQ: obs=<decimal>`. A build that returns always yields a result.

This runner fixes the COST problem. Booting the board is ~2-3 minutes and dominates a short
run, so testing four stages as four sessions is mostly boot time. The staged domains all
live in the same initramfs, so one boot can run them in sequence.

Ordering is load-bearing: stages ascend, and the FIRST one that fails to return is the
bisection point. Everything after it is lost, because a wedged domain takes the core with
it -- that is not a limitation to work around, it is the answer. Stop there and report.
"""
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fpga_driver import config as C
from fpga_driver.fpga_console import FpgaConsole
from fpga_driver.safe_cleanup import release_board, hard_exit, install_release_on_signal
from fpga_driver.run_ladder_perf_fpga import cold_boot, nvbit, install_resilient_emit
from fpga_driver.run_sqlite_baked_fpga import (
    IMG, IMG_NAME, BITSTREAM, assert_firmware_embeds_current_initramfs)

URL = os.environ.get("FPGA_URL")
HOST = os.environ.get("SQLITE_HOST") or "/test-domains/sqlite_host.user"
# Ascending stages. Overridable, but keep them ordered or the "first failure" logic lies.
DOMS = (os.environ.get("SQLITE_STAGE_DOMS") or
        "/test-domains/sqlite_stage0.dom,/test-domains/sqlite_stage1.dom,"
        "/test-domains/sqlite_stage2.dom,/test-domains/sqlite_stage3.dom").split(",")
# Per-domain budget. Short on purpose: a staged build that is going to return does so
# almost immediately, so silence here really does mean wedged -- unlike the full build,
# where silence can mean work in progress.
PER_DOM = float(os.environ.get("SQLITE_STAGE_TIMEOUT") or 90)
OUT = os.environ.get("PROBE_SCOPED_OUT") or "/tmp/capstone/sqlite-stages.txt"

STAGE_NAMES = {
    0: "entry+return only (shared region writable)",
    1: "after sqlite3_config(HEAP)  -- first touch of the 256 KB sqlite_heap",
    2: "after sqlite3_initialize()",
    3: "after sqlite3_open(:memory:) -- first real allocation traffic",
}


def log(m):
    print(f"[stages] {m}", file=sys.stderr, flush=True)


def decode(obs):
    """0x5A6E_ssrr -> (stage, rc), or None if this is not a staged marker."""
    if obs is None or (obs >> 16) != 0x5A6E:
        return None
    return (obs >> 8) & 0xff, obs & 0xff


def main():
    if not URL:
        raise SystemExit("FPGA_URL not set")
    assert_firmware_embeds_current_initramfs(IMG)

    console = FpgaConsole(URL, logger=lambda m: print(f"[fpga] {m}", file=sys.stderr))
    console.connect()
    install_resilient_emit(console)
    results, transcript = [], []
    try:
        console.lock()
        install_release_on_signal(console)
        rb = nvbit(console)
        if rb != BITSTREAM:
            raise SystemExit(f"HARD STOP: resident bitstream is {rb!r}, expected {BITSTREAM!r}")
        console.upload_boot_image(IMG_NAME, str(IMG))
        cold_boot(console, C.GDB_PROMPT, IMG_NAME)
        log(f"booted once; running {len(DOMS)} staged domains in sequence")

        for dom in DOMS:
            mark = console.uart_mark()
            wedged = False
            try:
                console.run_command(f"{HOST} {dom}; echo D''N_$?", r"DN_\d",
                                    timeout=PER_DOM, idle_timeout=PER_DOM)
            except Exception as exc:
                wedged = True
                log(f"{dom}: no return within {PER_DOM:.0f}s ({type(exc).__name__})")
            text = console.uart_since(mark)
            transcript.append(f"===== {dom} =====\n{text}\n")
            m = re.search(r"SQ: obs=(\d+)", text)
            obs = int(m.group(1)) if m else None
            results.append((dom, wedged, obs, "SQ: H/return" in text))
            if wedged:
                log("STOPPING: a wedged domain takes the core with it, so nothing after "
                    "this point would be meaningful")
                break

        pathlib.Path(OUT).write_text("".join(transcript))
        log(f"per-domain UART -> {OUT}")

        print("\n=== STAGED BISECTION ===", flush=True)
        first_bad = None
        for dom, wedged, obs, returned in results:
            d = decode(obs)
            if wedged:
                verdict = "WEDGED (no return)"
            elif d:
                verdict = f"returned stage={d[0]} rc={d[1]}"
            elif returned:
                verdict = f"returned, obs={obs} (not a staged marker)"
            else:
                verdict = f"no marker (obs={obs})"
            name = STAGE_NAMES.get(decode(obs)[0] if d else -1, "")
            print(f"  {dom:44} {verdict}{('   -- ' + name) if name else ''}", flush=True)
            if first_bad is None and (wedged or (d and d[1] != 0)):
                first_bad = (dom, wedged, d)

        if first_bad is None:
            # Deliberately does NOT name a stage. An earlier version said "the failure is
            # later than stage 3", which is only true for the default ascending ladder; run
            # with a PROBE set (stages 4/6/5) it printed a confident conclusion about
            # sqlite3_open that the run had not tested at all. State what was observed and
            # let the caller draw the boundary.
            ran = ", ".join(pathlib.Path(d).stem for d, _, _, _ in results)
            print(f"\nEvery domain in this set returned rc=0 ({ran}). The failure is "
                  f"outside what these stages cover -- widen or re-split.", flush=True)
        else:
            dom, wedged, d = first_bad
            if wedged:
                print(f"\nFIRST FAILURE: {dom} did not return. Everything below that stage "
                      f"works on silicon; the fault is inside that step.", flush=True)
            else:
                print(f"\nFIRST FAILURE: {dom} returned a nonzero SQLite rc {d[1]} at stage "
                      f"{d[0]} -- it did NOT wedge, so this is a normal SQLite error and "
                      f"the rc names it.", flush=True)
        return 0
    finally:
        print("RUN_DONE", flush=True)
        release_board(console, label="staged sqlite")


if __name__ == "__main__":
    hard_exit(main())
