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
from fpga_driver.fpga_console import FpgaConsole, ActionTimeout
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
# A wedged or entry-stalled domain emits TOTAL SILENCE, so waiting the whole per-domain
# budget for it is pure waste -- the call site used to pass the FULL budget, which made
# the idle check a no-op and burned the full timeout on a domain that died immediately. Any
# UART byte resets this clock, so a slow-but-live workload is unaffected; only silence trips
# it. Raise SQLITE_IDLE_S if a stage can legitimately go quiet for longer between SQ: markers.
IDLE_S = float(os.environ.get("SQLITE_IDLE_S") or 30)
OUT = os.environ.get("PROBE_SCOPED_OUT") or "/tmp/capstone/sqlite-stages.txt"
# Decimal `obs` values that are legitimate results even though they are not staged 0x5A6E
# markers -- i.e. sentinels a glue probe returns from a chosen bisection point.
#
# Without this the staged-marker guard hard-stops on the probe's own success value and throws
# away every remaining test in the boot. That happened on 2026-08-06: the precall probe returned
# 0x9E11 exactly as designed and the run aborted, wasting the slot that held the baseline. Board
# time is the scarce resource here, so a guard that cannot tell "unknown value" from "the value
# I asked for" costs a whole boot each time.
PROBE_SENTINELS = {int(v) for v in (os.environ.get("PROBE_SENTINELS") or "").replace(",", " ").split()}
# THE COMPLETE UART STREAM, exactly what the console GUI shows.
#
# OUT holds only per-test windows (`uart_since(mark)`), and a window CLOSES ON TIMEOUT while
# bytes are still arriving -- so a wedged test's capture can end mid-line and lose whatever the
# board emitted next. That produced a real disagreement between this runner's transcript and
# what a human watching the GUI saw, and a capture that disagrees with the board makes every
# verdict built on it unsafe. The console already accumulates everything; not writing it out was
# pure loss. Written in the FINALLY so a timeout, a wedge or a crash still leaves the evidence.
#
# Caveat kept from the skill: the console replays the PREVIOUS boot on connect (~548 KB), so
# scope any search to this run's own `load_image` rather than grepping the whole file.
RAW_OUT = os.environ.get("PROBE_RAW_OUT") or (OUT.rsplit(".", 1)[0] + "-raw.txt")

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


def decode_probe(obs):
    """0x5B_aa_bb_cc -> the S-07 operand-discrimination counters, or None.

    A SECOND sentinel exists because the probe reports an OBSERVATION, not a stage, and the
    two must never be confused: 0x5A6E_ssrr is "the ladder reached stage ss with rc rr",
    0x5B_aabbcc is "output_text saw aa untagged reloads, bb tagged offsets, cc persistent
    retries". Without this the runner HARD STOPs on a perfectly good measurement -- which it
    did on the first probe boot, discarding the run after the control arm had already proved
    the instrument works.

      aa  rs1_untagged      the reloaded capability read back NOT_CAP
      bb  rs2_tagged        the integer offset read back as something other than NOT_CAP
      cc  retry_persistent  re-ldc from the same slot was ALSO untagged  => memory
                            (cc == 0 with aa > 0 => the retry was tagged => register delivery)
    """
    if obs is None or (obs >> 24) != 0x5B:
        return None
    return (obs >> 16) & 0xff, (obs >> 8) & 0xff, obs & 0xff


def main():
    if not URL:
        raise SystemExit("FPGA_URL not set")
    # Verify the firmware against the domains THIS run will execute, not against the
    # canonical sqlite_silicon.dom -- a staged run's domains are named qr18/qr19/... and a
    # leftover file at the default path made the gate reject a perfectly fresh image.
    # The staged copy in overlay/test-domains is the right reference: finding its exact
    # bytes inside the decompressed initramfs proves "what I staged is what got packed".
    _repo = pathlib.Path(__file__).resolve().parents[4]
    _overlay = pathlib.Path(os.environ.get("SQLITE_STAGE_OVERLAY") or
                            _repo / "capstone/caplifive-system/sw/buildroot"
                                    "/overlay/test-domains")
    # A spec is "path", "path:selector", or -- under a ladder HOST override -- "rung:path",
    # where the .dom is the RIGHT half. Splitting unconditionally on the last ':' and keeping
    # the left half yields the rung NAME for every ladder entry, no such file exists in the
    # overlay, `_want` comes out EMPTY, and the freshness gate silently falls back to its
    # default (the canonical sqlite_silicon.dom). On 2026-08-10 that blocked a ladder-only
    # boot as "STALE FIRMWARE" over a leftover file the run did not reference -- the gate
    # being right about its question and wrong about its subject, which is precisely the
    # failure its own docstring records. Pick whichever side is actually a staged artifact.
    _want = []
    for spec in DOMS:
        tail = spec.split("|", 1)[-1]
        halves = tail.rsplit(":", 1)
        cands = [h for h in halves if (_overlay / pathlib.Path(h).name).is_file()] or [halves[0]]
        cand = _overlay / pathlib.Path(cands[-1]).name
        if cand.is_file():
            _want.append(cand)
    assert_firmware_embeds_current_initramfs(IMG, _want or None)

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
        log(f"booted once; running {len(DOMS)} staged domains in sequence")

        for dom_idx, dom_spec in enumerate(DOMS, 1):
            # "path" or "path:selector". The optional selector is passed to the host as its
            # second argument, which publishes it in the shared region so the DOMAIN picks the
            # probe at RUN TIME (see sqlite_capstone_domain.c, magic 0x5A6E00nn).
            #
            # This exists because the SHA5 entry stall is BUILD-DEPENDENT, not random: some
            # images enter reliably and others stall reliably (x101 6/6, r112 3/3), and the
            # runner stops at the first failure, so a stalling image at position 1 masks
            # everything behind it. Retrying a stalling binary is futile. Selecting the probe
            # at run time lets every measurement ride an image that is KNOWN to enter, instead
            # of drawing a fresh ticket per probe.
            # Optional per-entry HOST override: "host|path[:selector]".
            #
            # Without this every entry runs under HOST, which is the SQLite host; a ladder rung
            # needs the ladder controller and a different argv shape, so the two could never
            # share a boot. That mattered the moment a probe needed an INSTRUMENT VALIDATOR:
            # the trap-handler fault control is a ladder rung, and a control that rides a
            # DIFFERENT boot is not a control -- the board state it validates is the state that
            # died with the previous boot. Same reason the batching rule exists at all.
            if "|" in dom_spec:
                host, dom_spec = dom_spec.split("|", 1)
            else:
                host = HOST
            if ":" in dom_spec:
                dom, selector = dom_spec.rsplit(":", 1)
                host_args = f"{dom} {selector}"
            else:
                dom, selector = dom_spec, None
                host_args = dom
            label = f"{dom}:{selector}" if selector else dom
            # CLEAR THE TRAP LATCH BEFORE EACH DOMAIN, or the wedge read below is worthless.
            #
            # recent_nontrivial_*_log_q latches ANY trap except cause 0 (interrupt) and cause 2
            # (illegal instruction) -- cva6.sv:1077-1083 -- and is otherwise cleared only on
            # reset. mcause 9 is ECALL-from-S-mode, which OpenSBI and Linux emit constantly
            # during boot, so by the time a domain runs the latch is already set from normal
            # operation. On 2026-08-01 a wedge read returned 0x89 (seen=1, mcause=9) and was
            # very nearly reported as "the domain took an untrapped capability fault"; it was
            # a stale boot ecall. Capability faults are cause 23+code (24..28), not 9.
            #
            # cva6.sv:984: debug_byte_sel=3'b101 with debug_reg_sel=5'b11111, i.e.
            # switches = 0b10111111 = 191, clears the log. probe_wedge_regs.py:131-139 already
            # did this; the batch runner did not, which is what made the reading unattributable.
            try:
                for bit in range(8):
                    console.set_switch(bit, bool(191 & (1 << bit)))
                time.sleep(1.0)
                for bit in range(8):
                    console.set_switch(bit, False)
            except Exception as exc:  # never let instrumentation abort the run
                log(f"trap-log clear failed ({type(exc).__name__}) -- wedge reads will be stale")

            mark = console.uart_mark()
            wedged = False
            # NAME EVERY TEST ON THE UART, not just in this runner's local log.
            #
            # A single boot runs several domains back to back, so a human watching the board
            # console (or anyone reading a captured log later) otherwise sees an undifferentiated
            # stream of `SQ:` markers with no way to tell which domain produced them -- and when
            # one wedges, no way to tell WHICH one wedged without counting markers by hand.
            # Echoing the banner ON THE BOARD puts it in the UART stream itself, so it appears
            # live in the console GUI and inside `uart_since(mark)` (hence in the transcript).
            #
            # The prefix is `###`, deliberately NOT `SQ: `: the missing-domain guard below tests
            # `"SQ: " not in text`, so a banner carrying that string would make every run look
            # like it produced domain output and would silently disable the guard.
            n_tot = len(DOMS)
            start_banner = f"### TEST {dom_idx}/{n_tot} START {label} ###"
            log(f"--> TEST {dom_idx}/{n_tot}  {label}")
            t_dom = time.time()
            try:
                console.run_command(
                    f"echo '{start_banner}'; {host} {host_args}; rc=$?; "
                    f'echo "### TEST {dom_idx}/{n_tot} END {label} rc=$rc ###"; '
                    f"echo D''N_$rc",
                    r"DN_\d", timeout=PER_DOM, idle_timeout=IDLE_S)
                log(f"<-- TEST {dom_idx}/{n_tot}  {label}  returned in {time.time()-t_dom:.0f}s")
            except Exception as exc:
                wedged = True
                log(f"<-- TEST {dom_idx}/{n_tot}  {label}  NO RETURN within "
                    f"{PER_DOM:.0f}s ({type(exc).__name__}) -- everything after this is lost")
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
            # Check each entry against the marker ITS OWN host emits. The staged marker is
            # SQLite-specific; a ladder rung reports `RESULT <name> retval=<n>` and never emits
            # `SQ: obs=`, so applying the SQLite marker to a rung would hard-stop the session on
            # a perfectly good control. The guard stays strict for BOTH -- "the shell came back
            # but nothing ran" must not read as a pass, and that failure mode is identical for a
            # rung whose build silently staged nothing.
            is_sqlite = host == HOST
            m_obs = re.search(r"SQ: obs=(\d+)", text)
            m_rv = re.search(r"RESULT\s+\S+\s+retval=(-?\d+)", text)
            if is_sqlite:
                # 0x5A6E = the staged-dispatch marker family; 0x9Exx = the QUICKRET ladder
                # and every probe level built on it (0x9E33_LLrr for the workload ladder,
                # 0x9E26.. upward for the measurement probes). All are real results.
                #
                # MATCH ON THE FAMILY (top BYTE 0x9E), not on an enumerated value list. The
                # old check accepted only 0x5A6E/0x9E33 plus whatever was named in
                # PROBE_SENTINELS -- and PROBE_SENTINELS can only ever list the CORRECT
                # values. So a probe that returned a WRONG number, which is the entire point
                # of a measurement probe, hard-stopped the session and threw away every
                # remaining domain in the boot. That is exactly what happened: eight boots in
                # a row ran their control plus ONE probe and then aborted, because probe 1
                # returned the wrong value it was built to detect. `TEST 3/N` never appeared
                # in any of them. Cost: three slots of every four, and every level in the
                # matrix stuck at N=1.
                #
                # The guard's real job is "the shell came back but nothing ran", and the
                # family check still does that: an unstaged domain produces no `SQ: obs=` at
                # all, and a foreign marker still fails.
                _o = int(m_obs.group(1)) if m_obs else None
                bad = m_obs is None or ((_o >> 16) not in (0x5A6E,)
                                        and (_o >> 24) != 0x9E
                                        and _o not in PROBE_SENTINELS)
                got = "no SQ: obs= marker" if m_obs is None else f"obs={m_obs.group(1)}"
            else:
                bad = m_rv is None
                got = "no RESULT retval= marker"
            # INTERP_RETURN_PRECALL returns 0x9E11 from the glue, immediately before
            # domain_main. That is a DELIBERATE, meaningful result -- "the carve loop and
            # cap-init both completed" -- but it is not a staged 0x5A6E marker, so the guard
            # below used to call it "almost certainly was not staged" and hard-stop. On
            # 2026-08-08 it did exactly that to a valid run whose .dom had been verified
            # byte-present in rootfs.cpio before the boot, and whose UART showed the full
            # entry sequence (A/dom-ok, both shares, G/enter, ENT2:00009E11, H/return).
            # Recognise the sentinel instead of misreporting it as a staging failure.
            PRECALL_SENTINEL = 0x9E11
            if not wedged and bad and m_obs is not None and int(m_obs.group(1)) == PRECALL_SENTINEL:
                log(f"  {dom}: obs=0x9E11 -- INTERP_RETURN_PRECALL sentinel. The entry glue "
                    f"COMPLETED (carve loop AND cap-init); the fault is inside domain_main "
                    f"or in reaching it. This is a result, not a staging failure.")
                bad = False
            # A MISSING .dom IS DETECTED DIRECTLY, FROM THE LOADER'S OWN ERROR STRING.
            #
            # libcapstone.c:89,284 print exactly "Failed to open the file." when the domain
            # path does not exist. This is the ONLY reliable staging check, and it must be
            # tested BEFORE any marker-based reasoning, because the host does NOT stop there:
            # it goes on to create a domain, share both regions, enter and return, emitting a
            # complete and entirely convincing SQ: A/dom-ok .. G/enter .. H/return sequence
            # for a domain that was never loaded. A boot on 2026-08-10 reported four arms as
            # RETURNED on that basis when none of the four files existed in the initramfs.
            if not wedged and "Failed to open the file." in text:
                raise SystemExit(
                    f"HARD STOP: {dom} -- the loader printed 'Failed to open the file.'\n"
                    f"The domain is NOT in the initramfs. Every marker after that line "
                    f"(including SQ: G/enter and SQ: H/return) belongs to a domain that was "
                    f"never loaded, so any verdict from this arm is a PHANTOM.\n"
                    f"Re-stage the .dom into BOTH overlay/test-domains and "
                    f"build/target/test-domains, rebuild linux-rebuild THEN opensbi-rebuild, "
                    f"and verify the bytes are present in build/images/rootfs.cpio.")

            # A TRUNCATION ARM RETURNS WITHOUT EVER REACHING THE `obs=` EMIT.
            #
            # The guard's real job is "the shell came back but nothing ran". With the
            # missing-file case now caught above by the loader's own error string,
            # `SQ: G/enter` followed by `SQ: H/return` means the host entered a domain that
            # really was loaded and it came back.
            #
            # That shape is a byte-patched bisection arm: a 4-byte `j <epilogue>` planted
            # mid-function makes the domain return early, skipping the code that emits
            # `SQ: obs=`. Without this the guard aborts the whole boot at the first such
            # arm and discards every domain behind it -- which it did, killing a wide
            # 8-arm batch after arm 2 and costing a full boot.
            #
            # Staleness is a DIFFERENT failure and is covered elsewhere, by
            # assert_firmware_embeds_current_initramfs(), which checks the bytes actually
            # packed into the initramfs. This branch is not a substitute for it.
            if (not wedged and bad and m_obs is None
                    and "SQ: G/enter" in text and "SQ: H/return" in text):
                log(f"  {dom}: no obs= marker, but SQ: G/enter AND SQ: H/return are both "
                    f"present -- the domain ENTERED and RETURNED. Expected for a truncation "
                    f"arm that exits before the emit. This is a RETURN, not a staging failure.")
                bad = False
            # A PROBE REPORT IS A RESULT, NOT A STAGING FAILURE. 0x5B_aabbcc is the S-07
            # operand-discrimination counter; it is deliberately a different sentinel from the
            # ladder's 0x5A6E_ssrr because it reports an observation rather than a stage. The
            # first probe boot was thrown away here after its control arm had already proved the
            # instrument works -- the guard was right that it was not a STAGED marker and wrong
            # that it was not a marker.
            if not wedged and bad and decode_probe(m_obs) is not None:
                a, b, c = decode_probe(m_obs)
                log(f"  {dom}: S-07 probe report -- rs1_untagged={a} rs2_tagged={b} "
                    f"retry_persistent={c}")
                bad = False
            if not wedged and bad:
                raise SystemExit(
                    f"HARD STOP: {dom} produced {got}, not a staged marker.\n"
                    f"The domain almost certainly was not staged (a failed build stages "
                    f"nothing, and the host then exits 1, which matches the success regex).\n"
                    f"Verify the .dom exists in the overlay AND in the firmware before "
                    f"trusting any result from this session.\n"
                    f"(If you are running an INTERP_RETURN_PRECALL build, obs=40465 is the "
                    f"0x9E11 sentinel and is handled above -- this message means something else.)")

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
            transcript.append(f"===== {label} =====\n{text}\n")
            if is_sqlite:
                obs = int(m_obs.group(1)) if m_obs else None
                returned = "SQ: H/return" in text
            else:
                # A rung's retval IS its observation, and reaching the RESULT line is its
                # return. Recording None/False here would print a returning control as a
                # silent failure in the summary and invalidate every verdict behind it.
                obs = int(m_rv.group(1)) if m_rv else None
                returned = m_rv is not None
            # INFRASTRUCTURE vs DOMAIN wedge. `SQ: A/dom-ok` is printed by the host the
            # instant create_dom returns (sqlite_host.c), so its ABSENCE means the domain was
            # never created and NOTHING in it ran. `SPLB`/`SPLA` are monitor spin tags: 0xE006
            # is split_out_cap's unimplemented exact-fit case (sbi_capstone.c, guarded by
            # CAPSTONE_SPLIT_EXACT_FIT which is commented out), an M-mode `while(1)` that the
            # monitor's own comment records as wedging runs 5-7 in 4 of 4 boots and as the
            # source of "a large share of this campaign's random wedges".
            #
            # Without this distinction the summary below blames whichever domain happened to
            # occupy that slot. It did exactly that on 2026-08-06 and produced a confident,
            # entirely false localization of a SQLite function that never executed.
            created = "SQ: A/dom-ok" in text
            # ENTRY is a SEPARATE question from creation, and conflating them manufactures
            # false verdicts. `SQ: A/dom-ok` only means create_dom returned; the domain can
            # still fail to ENTER (R-16, the entry stall), in which case its markers stop at
            # `SHA5:` with no `SHA6:` and no `SQ: G/enter`, and NOTHING of the code under test
            # ran. This runner used to print "it WAS created and entered" on the strength of
            # A/dom-ok alone -- on 2026-08-06 it said exactly that about qr19b, whose domain
            # never entered (A/dom-ok=1, G/enter=0, SHA5=2, SHA6=1), which read as "this level
            # wedges" and nearly retracted a sound bisection. The board-run skill has always
            # keyed on `SQ: G/enter`; the runner did not.
            entered = "SQ: G/enter" in text
            montag  = re.search(r"(SPL[AB]|ILLX):([0-9A-Fa-f]{8})", text)
            results.append((label, wedged, obs, returned, created, entered,
                            montag.group(0) if montag else None))
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
                pc_bytes = {}
                mepc_bytes = {}
                try:
                    for sw, label, kind in ((255, "TRAP LOG {seen,mcause[6:0]}", "trap"),
                                            (224, "{excommit,ldsync,stsync,lsu_rdy,dyn_rdy,"
                                                  "flu_rdy,flush,privM}", "ready"),
                                            (225, "{tbe,wstore,wload,wrev,domsw,stall,memwr,"
                                                  "memwait}", "status"),
                                            # REV-NODE ALLOCATOR STATE. Every wedge so far
                                            # reads sw=225 = 0x95, i.e. wrev=1 AND memwait=1:
                                            # the dyn unit is blocked in
                                            # get_node_query_validity
                                            # (capstone_dyn_unit.anvil:106-112, a `recv` with no
                                            # abort path) while the rev-node unit is itself
                                            # waiting on the node-table memory read inside
                                            # get_rev_node (capstone_rev_node.anvil:36-41).
                                            # head/overflow/serving_idx say WHICH node id was
                                            # being queried and whether the bump allocator had
                                            # wrapped -- i.e. whether the id is plausible or
                                            # garbage, which separates "RTL drops a valid
                                            # query" from "we queried an unmappable id".
                                            (249, "rev_node_head[7:0]", "raw"),
                                            (250, "{overflow,0,head[9:8]}", "raw"),
                                            # serving_idx is 32 bits across 11011..11110
                                            # (cva6.sv:1186-1189). Reading only the low byte
                                            # cannot tell a legitimate node id from garbage,
                                            # which is exactly the discriminator needed:
                                            # a sane id (< head) means the hardware failed to
                                            # answer a VALID query; a huge/garbage id means we
                                            # queried an unmappable node and the RTL hung
                                            # instead of erroring.
                                            (251, "rev_node_serving_idx[7:0]", "raw"),
                                            (252, "rev_node_serving_idx[15:8]", "raw"),
                                            (253, "rev_node_serving_idx[23:16]", "raw"),
                                            (254, "rev_node_serving_idx[31:24]", "raw"),
                                            # COMMITTED pc, bytes 0..7 (reg_sel 0b00110..0b01101
                                            # == switches 230..237, probe_wedge_regs.py:30,64).
                                            # This runner never read it, which is why five board
                                            # sessions reported WHAT the core was stuck on
                                            # (wrev/memwait, mcause) and never WHERE. The last
                                            # committed pc maps straight onto the domain's
                                            # disassembly and names the faulting instruction --
                                            # the single most actionable number available at a
                                            # wedge, and it costs nothing extra because the board
                                            # is already held and powered for the reads above.
                                            #
                                            # Read it LAST: it is 8 switch round-trips, and if the
                                            # console drops out partway the cheap single-byte
                                            # diagnostics above are already captured.
                                            #
                                            # Treat 0xca11ab1ebadcab1e as NO DATA, not as a pc --
                                            # the debug path has twice returned that AXI
                                            # error-slave pattern.
                                            (230, "commit pc[7:0]", "pc"),
                                            (231, "commit pc[15:8]", "pc"),
                                            (232, "commit pc[23:16]", "pc"),
                                            (233, "commit pc[31:24]", "pc"),
                                            (234, "commit pc[39:32]", "pc"),
                                            (235, "commit pc[47:40]", "pc"),
                                            (236, "commit pc[55:48]", "pc"),
                                            (237, "commit pc[63:56]", "pc"),
                                            # LATCHED trap mepc, bank 3'b110 (sw = 192 + reg_sel).
                                            # This is the one that survives a wedge: the LIVE
                                            # `commit pc` bytes above read 0x2 once the core has
                                            # stopped committing, whereas these are latched at the
                                            # trap in the same block as the mcause. Added to the
                                            # RTL 2026-08-12; on an older bitstream these read
                                            # 0x00 and the assembled value is reported as absent
                                            # rather than as address 0.
                                            (196, "trap mepc[7:0]   (LATCHED)", "mepc"),
                                            (197, "trap mepc[15:8]  (LATCHED)", "mepc"),
                                            (198, "trap mepc[23:16] (LATCHED)", "mepc"),
                                            (199, "trap mepc[31:24] (LATCHED)", "mepc"),
                                            (200, "trap mepc[39:32] (LATCHED)", "mepc"),
                                            (201, "trap mepc[47:40] (LATCHED)", "mepc"),
                                            (202, "trap mepc[55:48] (LATCHED)", "mepc"),
                                            (203, "trap mepc[63:56] (LATCHED)", "mepc")):
                        for bit in range(8):
                            console.set_switch(bit, bool(sw & (1 << bit)))
                        time.sleep(1.2)
                        st = console.latest(C.LISTEN.get("led_state", "led_state"))
                        bits = st.get("states") if isinstance(st, dict) else None
                        v = sum((1 << i) for i, b in enumerate(bits) if b) if bits else None
                        line = (f"  [wedge] sw={sw:3} {label:52} "
                                f"{'UNREAD' if v is None else f'0x{v:02x} {v:08b}'}")
                        print(line, flush=True)
                        # Persist it. These readings were previously stdout-only, so the mcause
                        # and ready-bit values quoted in write-ups could not be checked against
                        # any artifact afterwards -- an audit flagged exactly that.
                        transcript.append(line + "\n")
                        if v is not None and kind == "ready":
                            names = ["privM", "flush", "flu_ready", "dyn_ready", "lsu_ready",
                                     "store_syncer", "load_syncer", "ex_commit.valid"]
                            print("          " + " ".join(f"{n}={(v >> i) & 1}"
                                                          for i, n in enumerate(names)),
                                  flush=True)
                        if kind == "pc":
                            pc_bytes[sw - 230] = v
                        if kind == "mepc":
                            mepc_bytes[sw - 196] = v
                    for bit in range(8):
                        console.set_switch(bit, False)

                    # Assemble the committed pc. Report it only when ALL EIGHT bytes were read:
                    # a partial read silently reconstructs a plausible-looking wrong address, and
                    # a wrong address sends the next session bisecting the wrong function.
                    if len(pc_bytes) == 8 and all(b is not None for b in pc_bytes.values()):
                        pc = sum(pc_bytes[i] << (8 * i) for i in range(8))
                        if pc == 0xca11ab1ebadcab1e:
                            print("  [wedge] commit pc = AXI ERROR-SLAVE PATTERN -- NO DATA, "
                                  "do not treat as an address", flush=True)
                        else:
                            print(f"  [wedge] commit pc = 0x{pc:016x}   <-- map this onto the "
                                  f"domain disassembly to name the faulting instruction",
                                  flush=True)
                    else:
                        missing = [i for i in range(8) if pc_bytes.get(i) is None]
                        print(f"  [wedge] commit pc UNREAD (missing bytes {missing}) -- "
                              f"reporting nothing rather than a partial address", flush=True)

                    # The LATCHED trap mepc. Same all-or-nothing rule as the pc above: a partial
                    # read reconstructs a plausible wrong address, and a wrong address sends the
                    # next session bisecting the wrong function.
                    if len(mepc_bytes) == 8 and all(b is not None for b in mepc_bytes.values()):
                        mepc = sum(mepc_bytes[i] << (8 * i) for i in range(8))
                        if mepc == 0xca11ab1ebadcab1e:
                            print("  [wedge] trap mepc = AXI ERROR-SLAVE PATTERN -- NO DATA",
                                  flush=True)
                        elif mepc == 0:
                            # Zero is what an older bitstream returns for an unimplemented mux
                            # slot, and it is also the reset value. Either way it is NOT evidence
                            # of a trap at address 0 -- say so instead of printing 0x0.
                            print("  [wedge] trap mepc = 0 -- either no trap was latched or this "
                                  "bitstream predates the mepc debug bank. NOT an address.",
                                  flush=True)
                        else:
                            print(f"  [wedge] trap mepc = 0x{mepc:016x}   <-- LATCHED at the "
                                  f"trap; map onto the domain disassembly to name the faulting "
                                  f"instruction", flush=True)
                    else:
                        missing = [i for i in range(8) if mepc_bytes.get(i) is None]
                        print(f"  [wedge] trap mepc UNREAD (missing bytes {missing})", flush=True)
                    # DO NOT CALL console.trace_dump() HERE, OR ANYWHERE ON A WEDGED CORE.
                    # Measured 2026-08-05: the trace dump HANGS THE BOARD HARD -- `trace_result`
                    # never arrives, the 60 s wait expires, and the board is left in a state that
                    # needs manual recovery. It cost a reflash. The capture that used to sit here
                    # has been removed for that reason; this comment is the guard rail, because
                    # `trace_dump()` is still present in fpga_console.py and reads like exactly
                    # the instrument a wedge investigation wants.
                except Exception as exc:
                    log(f"debug-mux read failed ({type(exc).__name__}) -- continuing to teardown")
                log("STOPPING: a wedged domain takes the core with it, so nothing after "
                    "this point would be meaningful")
                break

        pathlib.Path(OUT).write_text("".join(transcript))
        log(f"per-domain UART -> {OUT}")

        print("\n=== STAGED BISECTION ===", flush=True)
        first_bad = None
        for dom, wedged, obs, returned, created, entered, montag in results:
            d = decode(obs)
            if wedged:
                verdict = "WEDGED (no return)"
            elif d:
                verdict = f"returned stage={d[0]} rc={d[1]}"
            elif decode_probe(obs):
                a, b, c = decode_probe(obs)
                verdict = (f"S-07 PROBE: rs1_untagged={a} rs2_tagged={b} retry_persistent={c}"
                           + ("  [SELFTEST control fired]" if a == 0x40 and b == 0 and c == 0 else "")
                           + ("  => TAG LOST IN MEMORY" if a and c else "")
                           + ("  => REGISTER DELIVERY (retry was tagged)" if a and not c else "")
                           + ("  => rs2 GAINED A TAG; the untagged-capability reading is wrong here"
                              if b else ""))
            elif returned:
                verdict = f"returned, obs={obs} (not a staged marker)"
            else:
                verdict = f"no marker (obs={obs})"
            name = STAGE_NAMES.get(decode(obs)[0] if d else -1, "")
            if wedged and (not created or montag):
                verdict = ("INFRASTRUCTURE WEDGE (domain never created)" if not created
                           else f"INFRASTRUCTURE WEDGE (monitor tag {montag})")
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
                first_bad = (dom, wedged, d, created, entered, montag)

        if first_bad is None:
            # Deliberately does NOT name a stage. An earlier version said "the failure is
            # later than stage 3", which is only true for the default ascending ladder; run
            # with a PROBE set (stages 4/6/5) it printed a confident conclusion about
            # sqlite3_open that the run had not tested at all. State what was observed and
            # let the caller draw the boundary.
            # Report the ACTUAL values. An earlier version printed "returned rc=0" as fixed
            # text regardless of what came back -- it said rc=0 for a run whose rc was 177,
            # which is the wrong-but-confident output this project keeps being bitten by.
            got = ", ".join(
                f"{pathlib.Path(d).stem}=rc{decode(o)[1]}" if decode(o) else
                f"{pathlib.Path(d).stem}=?" for d, _, o, _, _, _, _ in results)
            print(f"\nEvery domain returned ({got}). No domain wedged; whether those values "
                  f"are CORRECT is the caller's judgement, not this runner's.", flush=True)
        else:
            dom, wedged, d, created, entered, montag = first_bad
            if wedged and (not created or not entered or montag):
                why = []
                if not created:
                    why.append("no `SQ: A/dom-ok` -- create_dom never returned, so the domain "
                               "was never created and NOTHING in it executed")
                elif not entered:
                    why.append("no `SQ: G/enter` -- the domain was CREATED but never ENTERED "
                               "(R-16 entry stall; markers stop at SHA5 with no SHA6), so "
                               "NOTHING of the code under test ran. R-16 is PER-IMAGE, so "
                               "retrying this binary is futile -- REDRAW: rebuild with a "
                               "harmless constant varied so the code under test is "
                               "byte-identical across draws, and sha256sum the set")
                if montag:
                    why.append(f"monitor spin tag {montag} -- the wedge is in M-mode, in the "
                               f"MONITOR, before/outside the domain")
                print(f"\nINFRASTRUCTURE WEDGE at {dom} -- THIS RUN CARRIES NO VERDICT "
                      f"ABOUT {dom}.\n  " + "\n  ".join(why) +
                      f"\n  Do NOT attribute this to the code under test. This spin is "
                      f"ordering/pool-state dependent: re-run with the sequence PERMUTED and "
                      f"a known-good domain in the same slot before concluding anything.",
                      flush=True)
            elif wedged:
                print(f"\nFIRST FAILURE: {dom} did not return, and it WAS created and entered "
                      f"(`SQ: A/dom-ok` AND `SQ: G/enter` present, no monitor tag). Everything "
                      f"below that stage works on silicon; the fault is inside that step.",
                      flush=True)
            else:
                print(f"\nFIRST FAILURE: {dom} returned a nonzero SQLite rc {d[1]} at stage "
                      f"{d[0]} -- it did NOT wedge, so this is a normal SQLite error and "
                      f"the rc names it.", flush=True)
        return 0
    finally:
        try:
            raw = getattr(console, "_uart", None)
            if raw:
                pathlib.Path(RAW_OUT).write_text(raw)
                print(f"[stages] FULL console stream ({len(raw)} bytes) -> {RAW_OUT}",
                      flush=True)
            else:
                print("[stages] no console buffer to dump", flush=True)
        except Exception as exc:
            print(f"[stages] raw dump failed ({type(exc).__name__})", flush=True)
        print("RUN_DONE", flush=True)
        release_board(console, label="staged sqlite")


if __name__ == "__main__":
    hard_exit(main())
