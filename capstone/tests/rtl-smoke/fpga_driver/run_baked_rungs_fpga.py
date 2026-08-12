"""Run silicon-ladder rungs BAKED INTO THE FIRMWARE IMAGE, not shipped over UART.

Why this exists. `run_ladder_perf_fpga.py` transfers the controller and every `.dom` over the
console as gzip+base64 at 16 chars per emit, and each emit is an HTTPS round trip -- minutes per
domain, and it dominated every sweep. These rungs are ~10 KB each, so putting them in the
initramfs costs nothing on the JTAG upload that happens anyway, and the run becomes a shell
command per rung.

R-3 STILL APPLIES HERE -- do not drop the relocation. An earlier version of this docstring
claimed invoking from a shell "removes the R-3 confound". It does not. R-3 is a MISSING ICACHE
INVALIDATE on the domain switch (ISSUES.md), so a domain reused at entry VA 0x10000 within one
boot silently hangs its cscall; the monitor still lacks that invalidate. This runner calls
cold_boot ONCE and then runs every rung from that single boot, which is exactly the R-3
condition. Invoking from a shell adds no invalidate, and RECORDING position does not REMOVE a
confound. Build each rung at a distinct DOMAIN_BASE_VA (0x10000, 0x20000, ...) or expect a
silent hang that looks just like a rung result. Position is still reported -- see `pos=`.

Usage:
    BAKED_RUNGS="clp1 clp2 clp4 clp8" python3 -m fpga_driver.run_baked_rungs_fpga
Env: FPGA_URL, FPGA_FW (both required, no defaults -- an implicit FW silently boots whatever
was built last), BAKED_CTL (default /test-domains/lpc), BAKED_TIMEOUT (default 120).
"""
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fpga_driver import config as C                                    # noqa: E402
from fpga_driver.fpga_console import FpgaConsole, ActionTimeout        # noqa: E402
from fpga_driver.safe_cleanup import (release_board, hard_exit,        # noqa: E402
                                      install_release_on_signal)
from fpga_driver.run_sqlite_baked_fpga import _hash_name  # noqa: E402
from fpga_driver.run_ladder_perf_fpga import (cold_boot, nvbit,        # noqa: E402
                                              install_resilient_emit)

URL = os.environ.get("FPGA_URL")
if not URL:
    raise SystemExit("FPGA_URL not set")
IMG = pathlib.Path(os.environ["FPGA_FW"])          # KeyError on purpose: never default the FW
# NAME THE STORED IMAGE BY CONTENT, never by a fixed filename.
#
# This used to default to "fw_payload_fpga_up_gpfree.bin" -- a single slot that every run
# overwrote. Two consequences, both bad for an A/B: the board's image store cannot tell two
# firmwares apart, and the reassuring "19563528 bytes written" line proves nothing about WHICH
# firmware reached DDR, because that byte count was identical across every build on 2026-08-05.
# A content hash makes the loaded image self-identifying and lets an unchanged firmware reuse
# the stored upload instead of re-sending ~19 MB. This is the stale-store trap cold_boot's own
# docstring warns about, and the other runners already avoid it.
IMG_NAME = os.environ.get("FPGA_FW_NAME") or _hash_name(IMG)
CTL = os.environ.get("BAKED_CTL") or "/test-domains/lpc"
RUNGS = (os.environ.get("BAKED_RUNGS") or "clp1 clp8").split()
TIMEOUT = float(os.environ.get("BAKED_TIMEOUT") or 120)
# A wedged domain emits TOTAL SILENCE, so waiting the full run budget for it is
# pure waste -- callers used to pass idle_timeout=TIMEOUT, which made the idle
# check a no-op and spent the whole 150s on a domain that died in the first
# millisecond. Any UART byte resets this clock, so a slow-but-live workload is
# unaffected; only silence trips it. Raise BAKED_IDLE_S if a workload can legitimately
# go quiet for longer than this between markers.
IDLE_S = float(os.environ.get("BAKED_IDLE_S", "25"))
ART = pathlib.Path(os.environ.get("LADDER_FPGA_DIR") or "/tmp/capstone/ladder-fpga")
# THE COMPLETE UART STREAM. Ported from run_sqlite_stages_fpga.py, which explains why the
# per-test windows alone are unsafe: `uart_since(mark)` CLOSES ON TIMEOUT while bytes are still
# arriving, so a wedged rung's capture can end mid-line. The staged runner grew a raw capture
# for that reason; this one never did, and the consequence was concrete -- boots 31 through 43
# have no full console capture at all, and those are exactly the boots whose SHA5-vs-SHA6
# classification decided whether seven probes had entered. That classification survived only
# because the driver's stderr happened to be redirected to a scratch file. Luck, not design.
RAW_OUT = os.environ.get("BAKED_RAW_OUT") or (
    (os.environ.get("BAKED_OUT") or "/tmp/capstone/baked-rungs.txt").rsplit(".", 1)[0] + "-raw.txt")
OUT = os.environ.get("BAKED_OUT") or "/tmp/capstone/baked-rungs.txt"
# Resident-bitstream guard. Reflashed 2026-08-04 to caplifive_fixed_forward.bit, which
# carries the operand-forwarding fix (capstone-ariane 7aac52f93). Overridable so the next
# reflash needs no code change mid-session -- the guard exists to stop a run from silently
# measuring the WRONG silicon, and on 2026-08-04 it did exactly that.
BITSTREAM = os.environ.get("FPGA_BITSTREAM", "caplifive_fixed_forward.bit")


def log(m):
    print(f"[baked] {m}", file=sys.stderr, flush=True)


def main():
    oracles = {}
    for r in RUNGS:
        p = ART / f"{r}.oracle"
        if not p.is_file():
            raise SystemExit(f"missing oracle {p} -- run build-ladder-fpga.sh {r}")
        oracles[r] = int(p.read_text().split()[0])

    console = FpgaConsole(URL, logger=lambda m: print(f"[fpga] {m}", file=sys.stderr))
    console.connect()
    install_resilient_emit(console)
    results, transcript = [], []
    try:
        console.lock()
        install_release_on_signal(console)
        rb = nvbit(console)
        # STRICT equality, and None must HARD STOP. The old form was
        # `if rb and BITSTREAM not in rb`, which had two holes: nvbit() polls only 8s and
        # returns None on timeout, so an unreadable flash_state SKIPPED the guard entirely; and
        # a substring test lets `<name>_v2.bit` satisfy a guard set to `<name>.bit`. For an
        # experiment whose entire independent variable IS the bitstream, either hole means
        # silently measuring the wrong silicon. The other two runners already use `!=`.
        if rb != BITSTREAM:
            raise SystemExit(
                f"HARD STOP: resident bitstream is {rb!r}, expected {BITSTREAM!r}. "
                f"(None means flash_state was unreadable within nvbit()'s poll -- treat as "
                f"unknown silicon, never as a pass.)")
        console.upload_boot_image(IMG_NAME, str(IMG))
        # Boot is ONE RETRYABLE UNIT. Measured 2026-08-13 across 15 board runs: 2 failed in
        # cold_boot BEFORE a single byte of the image moved, with abstractcs busy stuck at 1 --
        # the hart intermittently fails to enter debug mode on `reset halt`. Both were
        # recoverable by retry, and run_ladder_perf_fpga.py has carried exactly this loop since
        # 26-07 for the same reason; this call site simply never got it, so a transient
        # infrastructure failure aborted the whole session and cost a boot.
        #
        # It is NOT image size: the failures transferred ZERO bytes, and 12 successful loads of
        # the same firmware took 131.99-133.25 s against a 300 s budget. cold_boot is idempotent
        # and self-cleaning (its own `finally: gdb_stop()`), so a retry costs a reload.
        for _boot_attempt in range(1, 4):
            try:
                cold_boot(console, C.GDB_PROMPT, IMG_NAME)
                break
            except (ActionTimeout, RuntimeError) as _e:
                log(f"  cold_boot failed ({_e}); retry {_boot_attempt}/3")
                if _boot_attempt == 3:
                    raise
        log(f"booted; running {len(RUNGS)} baked rungs from the shell")

        for pos, r in enumerate(RUNGS, 1):
            mark = console.uart_mark()
            banner = f"### RUNG {pos}/{len(RUNGS)} {r} ###"
            try:
                console.run_command(
                    f"echo '{banner}'; {CTL} {r} /test-domains/{r}.dom; "
                    f"rc=$?; echo \"### RUNG {pos} {r} END rc=$rc ###\"; echo D''N_$rc",
                    r"DN_\d", timeout=TIMEOUT, idle_timeout=IDLE_S)
                wedged = False
            except Exception as exc:
                wedged = True
                log(f"{r}: no return within {TIMEOUT:.0f}s ({type(exc).__name__})")
            text = console.uart_since(mark)
            transcript.append(f"===== {r} (pos {pos}) =====\n{text}\n")
            m = re.search(r"RESULT\s+" + re.escape(r) + r"\s+retval=(\d+)", text)
            got = int(m.group(1)) if m else None
            # ENTERED vs merely CREATED. A ladder rung that entered emits SHA6; absence means it
            # died in the first region share and NOTHING of the code under test ran. Printing
            # "NO RESULT (wedged)" for both is what let seven consecutive entry stalls read as
            # seven negative results about the code -- see the staged runner, which was fixed
            # for the same confusion on the SQLite path.
            entered = "SHA6:" in text
            results.append((r, pos, got, oracles[r]))
            if got is None and not entered:
                log(f"  {r}: NO SHA6 -- ENTRY STALL (R-16), the domain never ran. This slot "
                    f"carries NO VERDICT about the code. R-16 is per-image: REDRAW (vary "
                    f"FDREG_DRAW/QR_DRAW, sha256sum the set), do not retry this binary.")
            else:
                log(f"  {r}: retval={got} oracle={oracles[r]} "
                    f"{'OK' if got == oracles[r] else 'MISMATCH/NO-RESULT'}"
                    f"{'' if entered else ' [no SHA6 -- did not enter]'}")
            if wedged:
                # A WEDGED DOMAIN TAKES THE CORE. Everything after this point is collateral,
                # not a result, and must not be scored -- this runner used to continue and emit
                # five independent-looking NO-RESULT rows for what was ONE failure. R-3's own
                # entry records the cost: "rv8_primes hung and the runner kept 'reusing' the
                # dead boot, losing the four rungs after it -- all of which had worked minutes
                # earlier". Stop issuing commands and label the remainder honestly.
                for rem_pos, rem in enumerate(RUNGS[pos:], pos + 1):
                    results.append((rem, rem_pos, "COLLATERAL", oracles[rem]))
                log(f"STOPPING after {r} wedged: {len(RUNGS) - pos} later rung(s) are "
                    f"collateral, not results")
                break
    finally:
        print("RUN_DONE", flush=True)
        try:
            release_board(console, label="baked rungs")
        except Exception as exc:
            print("release warn:", exc, flush=True)
        # GUARDED: this used to be bare, and it runs BEFORE the summary print. A failure here
        # (missing /tmp/capstone, full disk) raised inside the finally and threw away results
        # that had already been captured.
        try:
            # The FULL console stream, written first and separately: it is the only capture that
            # survives a window closing on timeout, and it is what the SHA6 entry check must be
            # read from. Written in the finally so a wedge, a timeout or a crash still leaves it.
            # FpgaConsole exposes `uart_text` (fpga_console.py:472). It has never had `uart_all`
            # or `uart_buf`, so this silently wrote nothing for 19 consecutive boots -- including
            # every stage-34 entry stall, which is exactly what this capture exists to diagnose.
            # getattr-with-default hid the error; name the attribute directly so a rename breaks
            # loudly instead of quietly.
            raw = getattr(console, "uart_text", "")
            if callable(raw):
                raw = raw()
            if raw:
                pathlib.Path(RAW_OUT).write_text(raw if isinstance(raw, str) else raw.decode("utf-8", "replace"))
                print(f"full console stream -> {RAW_OUT}", flush=True)
        except Exception as exc:
            print(f"raw capture failed ({type(exc).__name__}) -- windows still written",
                  file=sys.stderr, flush=True)
        try:
            pathlib.Path(OUT).write_text("".join(transcript))
        except Exception as exc:
            print(f"transcript write failed ({type(exc).__name__}) -- summary still follows",
                  flush=True)
        print(f"\n==== BAKED RUNGS (one boot, {len(results)} rungs) ====", flush=True)
        print(f"{'rung':10} {'pos':>4} {'retval':>8} {'oracle':>8}  verdict", flush=True)
        for r, pos, got, orc in results:
            if got == "COLLATERAL":
                v = "COLLATERAL (after a wedge -- NOT a result)"
            elif got is None:
                v = "NO RESULT (wedged)"
            elif got == orc:
                v = "OK"
            else:
                v = "WRONG"
            print(f"{r:10} {pos:>4} {str(got):>8} {orc:>8}  {v}", flush=True)
        print(f"transcript -> {OUT}", flush=True)
    # EXIT STATUS MUST REFLECT THE RUN. This was hard_exit(0) unconditionally, so a session in
    # which every rung wedged exited 0 and any wrapper keying off status read it as success.
    ok = bool(results) and all(g == o for _, _, g, o in results)
    hard_exit(0 if ok else 1)


if __name__ == "__main__":
    main()
