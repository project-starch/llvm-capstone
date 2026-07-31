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
import time
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

            # A MISSING DOMAIN MUST NOT READ AS SUCCESS.
            #
            # `sh` answers a nonexistent path with "not found" and exit 127, so the echo
            # prints DN_127 -- which MATCHES the r"DN_\d" success pattern. Without this
            # check the domain is recorded as having run, `obs` is None, the first-bad test
            # never fires, and the summary prints "every domain returned rc=0", i.e. a
            # confident pass from a session that executed nothing. That is the same class
            # of failure as the 2026-07-30 stale-initramfs incident (exit 127 read as a
            # domain failure), and it is what currently makes pruning the overlay unsafe:
            # today nothing is ever deleted, so a locally-present domain is necessarily in
            # the firmware, and that accident is the only thing masking this hole.
            # A DOMAIN THAT WAS NEVER STAGED MUST NOT READ AS SUCCESS EITHER.
            #
            # The exit-127 check below catches the SHELL failing to find the host binary. It
            # does NOT catch the far more common case: the host binary exists and runs, the
            # .dom does not, so the host reports its own failure and exits 1. DN_1 matches
            # r"DN_\d", so the domain was recorded as having returned, and with no staged
            # marker in its output the summary printed "Every domain in this set returned
            # rc=0" -- a clean false pass. Measured 2026-07-31: five domains whose builds had
            # ALL failed, staged nothing, and the run reported success.
            #
            # So: a run that produced no `SQ: obs=` marker at all, or a marker that is not a
            # staged marker, is a HARD failure. A domain that actually ran always emits one.
            # ONLY when the domain RETURNED. A wedged domain legitimately produces no marker,
            # and an earlier version of this check ran unconditionally -- so it hard-stopped on
            # every genuine wedge, which is the case it is supposed to let through, and
            # suppressed the in-session debug-mux read below. The check is for "the shell came
            # back but nothing ran", not for "the core died".
            m_obs = re.search(r"SQ: obs=(\d+)", text)
            if not wedged and (m_obs is None or (int(m_obs.group(1)) >> 16) != 0x5A6E):
                got = "no SQ: obs= marker" if m_obs is None else f"obs={m_obs.group(1)}"
                raise SystemExit(
                    f"HARD STOP: {dom} produced {got}, not a staged marker.\n"
                    f"The domain almost certainly was not staged (a failed build stages "
                    f"nothing, and the host then exits 1, which matches the success regex).\n"
                    f"Verify the .dom exists in the overlay AND in the firmware before "
                    f"trusting any result from this session.")

            m_rc = re.search(r"DN_(\d+)", text)
            if m_rc and int(m_rc.group(1)) == 127:
                raise SystemExit(
                    f"HARD STOP: {dom} is NOT PRESENT on the board (exit 127).\n"
                    f"The firmware does not carry it -- re-stage and relink, do not trust "
                    f"any result from this session.")
            if "not found" in text and "SQ: " not in text:
                raise SystemExit(
                    f"HARD STOP: {dom} produced no domain output and the shell reported "
                    f"'not found'. Treating this as a pass would test nothing.")
            transcript.append(f"===== {dom} =====\n{text}\n")
            m = re.search(r"SQ: obs=(\d+)", text)
            obs = int(m.group(1)) if m else None
            results.append((dom, wedged, obs, "SQ: H/return" in text))
            if wedged:
                # INSTRUMENT THE WEDGE HERE, IN THIS SESSION.
                #
                # The core is wedged RIGHT NOW, with the lock held, the board powered and the
                # console live. Reading the debug mux costs ~20 s. Doing it in a separate
                # session costs a full boot -- upload, JTAG load, kernel, initramfs -- roughly
                # 200 s, and re-creates the state by re-running rather than observing the
                # state that actually failed. Every wedge investigated this way so far paid
                # that cost for no reason.
                #
                # Selectors verified against cva6.sv:1090-1215; byte_sel must be 0b111, so the
                # switch value is 224 + reg_sel. Decoded by name because a raw hex byte has
                # been misread twice (0x84 and 0x89 both were).
                log("WEDGED -- reading the debug mux now, before releasing the board")
                try:
                    for sw, label, kind in ((255, "TRAP LOG {seen,mcause[6:0]}", "trap"),
                                            (224, "{excommit,ldsync,stsync,lsu_rdy,dyn_rdy,"
                                                  "flu_rdy,flush,privM}", "ready"),
                                            (225, "{tbe,wstore,wload,wrev,domsw,stall,memwr,"
                                                  "memwait}", "status")):
                        for bit in range(8):
                            console.set_switch(bit, bool(sw & (1 << bit)))
                        time.sleep(1.2)
                        st = console.latest(C.LISTEN.get("led_state", "led_state"))
                        bits = st.get("states") if isinstance(st, dict) else None
                        v = sum((1 << i) for i, b in enumerate(bits) if b) if bits else None
                        print(f"  [wedge] sw={sw:3} {label:52} "
                              f"{'UNREAD' if v is None else f'0x{v:02x} {v:08b}'}", flush=True)
                        if v is not None and kind == "ready":
                            names = ["privM", "flush", "flu_ready", "dyn_ready", "lsu_ready",
                                     "store_syncer", "load_syncer", "ex_commit.valid"]
                            print("          " + " ".join(f"{n}={(v >> i) & 1}"
                                                          for i, n in enumerate(names)),
                                  flush=True)
                    for bit in range(8):
                        console.set_switch(bit, False)
                except Exception as exc:
                    log(f"debug-mux read failed ({type(exc).__name__}) -- continuing to teardown")
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
            # A NON-ZERO rc IS NOT A FAILURE. Only a WEDGE is.
            #
            # This rule used to score any non-zero rc as the first failure and stop the
            # ladder. That is wrong for every probe whose SUCCESS value is non-zero -- the
            # h30..h34 holder ladder returns 40/100/160/255, the byte-survival probes return
            # 255, the watchdog markers are 0xB1..0xB6, and c14fan returns 50 or 55. It
            # printed "FIRST FAILURE: h30 returned a nonzero rc 40" for a correct result and
            # let h32's genuine rc=0 mismatch pass unflagged, and it would abort any batch
            # after the first watchdog marker.
            #
            # Expected values are the caller's business, not this runner's: it cannot know
            # what a given stage should return. So it reports what came back and flags only
            # what it can judge on its own -- a domain that never returned.
            if first_bad is None and wedged:
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
