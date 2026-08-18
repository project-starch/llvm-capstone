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
import itertools
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
# Which domains get the trap-log clear (switch 191) before they run.
#
#   all    (default, and the historical behaviour) -- clear before every domain
#   first  -- clear once, before the first domain only
#   none   -- never clear
#
# WHY IT IS WORTH TURNING OFF FOR THE DOMAIN YOU EXPECT TO WEDGE. 191 is also a
# BLIND WINDOW: the logging always_ff is `if (clear) <clear> else <record>`, so
# while the switches sit there no displacement is recorded, and a missed
# displacement looks exactly like case (b) ("memory did it"). Skipping the clear
# for the run under test gives that run a ZERO blind window, which is what turns
# an all-zero S-07 byte from "maybe we just didn't see it" into a real negative.
#
# What you give up is small and recoverable: the clear exists so a STALE latch
# cannot be read as this domain's trap. But the trap latch is LAST-WRITER-WINS --
# recent_nontrivial_{mcause,mepc} are overwritten on every non-trivial trap -- so
# the wedge's own mcause-25 still lands on top of whatever a previous domain left.
# The only lost distinction is "this domain trapped" vs "a previous domain trapped
# and this one wedged without latching one", and the per-domain trap-summary
# sample below closes that: if the post-run latch is identical to the pre-run
# value, the trap fields are stale and carry no verdict. The displacement sticky
# is unaffected either way -- 191 does not clear it.
TRAPLOG_CLEAR = (os.environ.get("SQLITE_TRAPLOG_CLEAR") or "all").strip().lower()
if TRAPLOG_CLEAR not in ("all", "first", "none"):
    raise SystemExit(f"SQLITE_TRAPLOG_CLEAR must be all|first|none, got {TRAPLOG_CLEAR!r}")
OUT = os.environ.get("PROBE_SCOPED_OUT") or "/tmp/capstone/sqlite-stages.txt"

# ---------------------------------------------------------------------------------
# DESTRUCTIVE SWITCH APERTURES -- why the driver must choose its own transit path.
#
# The debug mux is addressed by the 8 DIP switches, and a few apertures are not passive
# reads: on the gen-3 probe, 220 is the SELFTEST TRIGGER, which injects a synthetic LDC
# record. Landing on it -- even in passing -- destroys the record the boot exists to read.
#
# THE DWELL COUNTER DOES NOT PROTECT AGAINST THIS. ~21 ms of required stability defeats
# contact BOUNCE, which is what it is for. It does not defeat a slow TRANSIT: a value that
# is merely passed through is held for the same duration class as one deliberately applied,
# so it fires identically. Time cannot fix it; only ORDER can.
#
# And this driver was doing exactly the destructive walk. `_read_sw` flipped bits in
# ascending order, so 221 -> 222 cleared bit0 first:
#
#     221 = 0b11011101  ->  0b11011100 = 220  ->  0b11011110 = 222
#
# with each `set_switch` an HTTPS round trip, i.e. the intermediate is held for far longer
# than the dwell. The safe walk for that pair is 221 -> 223 -> 222; 223 is ldc_pc[23:16],
# a harmless read aperture.
#
# The rule is encoded here rather than written down for a human, because the driver is what
# will still be doing this in a month. A transition with no safe ordering raises instead of
# guessing -- there is no silent fallback, since the whole point is that landing on the
# trigger is unrecoverable for that boot.
DESTRUCTIVE_SWITCHES = frozenset({220})

# LED PULSE-STRETCHER SETTLE. Each LED bit is held high for 2^20 core cycles (~21 ms at
# 50 MHz) after it was last driven, so a reading taken sooner is an OR across apertures.
# 0.5 s is ~24 windows -- generous on purpose, since the cost is seconds per sample and the
# failure mode is a decoded verdict that looks entirely plausible.
# BOTH S-07 RECORDS ROLL. THEY ARE NOT ONE-SHOTS, AND THIS DRIVER USED TO SAY THEY WERE.
#
# Verified against the RTL that is actually flashed (capstone-ariane 6882b265f, a descendant of
# 8c75d899b "Withdraw the gen-3 probe: keep only the rolling record"):
#
#   core/load_unit.sv:774   if (ldc_result_back && !req_port_i.data_rtag) begin ...
#   core/store_unit.sv:549  if (store_buffer_valid && st_is_cap_q) begin ...
#
# Neither condition carries a `!..._valid_q` guard, and the load_unit comment says so in as many
# words: "capture EVERY LDC response that comes back untagged, so the record holds the most
# recent one and the wedge is what freezes it."
#
# The driver's "PROBE ALREADY SPENT / carries NO weight" messages were written for the one-shot
# design and are FALSE here. They caused boot 4 (2026-08-18) to dismiss its own 208 readings.
# ldc0_valid=1 means only "at least one untagged LDC has happened since reset", which is routine
# -- boot software produces them from miss refills. It does NOT mean the probe is used up.
#
# What is TRUE, and is the real limit on this bitstream: gen 1 has no 193/194 register-level
# correlation gate (193 is {5'b0,store_buf_commit_cnt}), so a granule match between the two
# rolling records is SUGGESTIVE and UNLICENSED -- either record may have rolled onto that granule
# independently. Report it as correlated-by-granule, never as proven tag loss.
S07_RECORDS_ROLL = (os.environ.get("S07_RECORDS_ROLL") or "1") == "1"

LED_SETTLE_S = float(os.environ.get("LED_SETTLE_S") or 0.5)
# How long to wait for a FRESH led_state payload before falling back to the cached one.
LED_FRESH_TIMEOUT_S = float(os.environ.get("LED_FRESH_TIMEOUT_S") or 5.0)


def safe_switch_bit_order(cur, target, forbidden=DESTRUCTIVE_SWITCHES):
    """Return an order for the differing bits such that no INTERMEDIATE value is forbidden.

    Endpoints are the caller's business; only values strictly between are checked."""
    if target in forbidden:
        raise ValueError(f"switch {target} is a destructive aperture and must not be read")
    diff = [b for b in range(8) if ((cur ^ target) >> b) & 1]
    if not diff:
        return []
    for order in itertools.permutations(diff):
        v, ok = cur, True
        for b in order[:-1]:              # the last flip lands on target, checked above
            v ^= (1 << b)
            if v in forbidden:
                ok = False
                break
        if ok:
            return list(order)
    raise RuntimeError(
        f"no safe switch path from {cur} to {target} avoiding {sorted(forbidden)}")


# POSITIVE CONTROL, run at import. A path-chooser that has never rejected anything is not a
# working path-chooser, so this asserts BOTH that the naive ascending order really does land
# on the trigger (the bug this replaces) and that the chosen order does not.
def _selftest_switch_path():
    naive, v = [], 221
    for b in range(8):
        if ((221 ^ 222) >> b) & 1:
            v ^= (1 << b)
            naive.append(v)
    assert 220 in naive, "expected the naive ascending walk 221->222 to pass through 220"
    v, seen = 221, []
    for b in safe_switch_bit_order(221, 222):
        v ^= (1 << b)
        seen.append(v)
    assert 220 not in seen, f"safe walk still hit the trigger: {seen}"
    assert seen[-1] == 222 and seen[0] == 223, f"expected 221->223->222, got {seen}"


_selftest_switch_path()


# The physical switch value the board is holding, as far as this process knows. None means
# "unknown", which is the state at connect: the board keeps whatever the last run left.
_SW_CURRENT = None


def set_switch_value(console, target):
    """Drive the switches to `target` without ever RESTING on a destructive aperture.

    From an unknown state the value is first walked to 0 by clearing bit 7 FIRST. Every
    forbidden aperture here has bit 7 set, so once bit 7 is low no subsequent intermediate
    can equal one, whatever the unknown starting value was."""
    global _SW_CURRENT
    assert all((f >> 7) & 1 for f in DESTRUCTIVE_SWITCHES), (
        "the bit-7-first argument below assumes every forbidden value has bit 7 set")
    if _SW_CURRENT is None:
        for bit in [7] + [b for b in range(8) if b != 7]:
            console.set_switch(bit, False)
        _SW_CURRENT = 0
    for bit in safe_switch_bit_order(_SW_CURRENT, target):
        _SW_CURRENT ^= (1 << bit)
        console.set_switch(bit, bool(target & (1 << bit)))
    _SW_CURRENT = target
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


def decode_s07_retry(obs):
    """0x5C_ss_pp_rr -> (saw_untagged, retry_still_untagged, retry_recovered), or None.

    The discriminator the RTL lane asked for after refuting the syncer-displacement cause:
      rr > 0  the retry from the SAME address came back TAGGED -> memory was never wrong,
              so the fault is in REGISTER DELIVERY (the LOAD_WB-erases-the-capability path
              they confirmed in simulation);
      pp > 0  the retry was ALSO untagged -> memory genuinely lost the tag, which is the
              shadow-tag refill path (A-2), so far unprobed.
    """
    if obs is None or (obs >> 24) not in (0x5C, 0x5D, 0x5E):
        return None
    return (obs >> 16) & 0xff, (obs >> 8) & 0xff, obs & 0xff


def assemble_mepc(mepc_bytes, probe_gen2):
    """Assemble the latched mepc, correctly for BOTH probe generations.

    THIS IS THE SILENT-GARBAGE HAZARD ON THIS DRIVER. Up to and including the s07tag2
    bitstream, switches 196..203 were mepc[7:0]..mepc[63:56] and an eight-byte assembly was
    right. In the next generation, 201/202/203 are RECLAIMED for stc_pc[23:0], so the same
    eight-byte assembly silently produces a plausible wrong address -- no error, no missing
    byte, just a number that names the wrong instruction. Every localisation this campaign has
    made rests on that number.

    The reclaim is information-lossless: this is an sv39 core, so mepc[63:39] is pure sign
    extension of bit 38, and mepc[39:0] (switches 196..200) covers every VA the core can
    generate. So generation 2 assembles five bytes and sign-extends from bit 38.

    Returns (value, note) or (None, reason).
    """
    need = 5 if probe_gen2 else 8
    have = [mepc_bytes.get(i) for i in range(need)]
    if any(b is None for b in have):
        missing = [i for i in range(need) if mepc_bytes.get(i) is None]
        return None, f"UNREAD (missing bytes {missing})"
    raw = sum(b << (8 * i) for i, b in enumerate(have))
    if not probe_gen2:
        return raw, "8-byte assembly (probe gen 1)"
    # sign-extend from bit 38
    val = raw & ((1 << 40) - 1)
    if val & (1 << 38):
        val |= ~((1 << 39) - 1) & ((1 << 64) - 1)
    return val, "5-byte assembly + sign-extension from bit 38 (probe gen 2; 201-203 are stc_pc)"


def decode_probe_generation(b193):
    """True if this bitstream carries the gen-2 probe. KEYED ON SWITCH 193, NOT 216.

    NOT 216, and the reason is the whole point. Bit 7 of 216 is a hardwired 1 in gen 2, so
    bit7==0 there soundly means "no census in this bitstream". The CONVERSE does not hold, and
    the converse is what a discriminator relies on: on the CURRENT bitstream reg 24 is not a
    default arm, it is tval[47:40] -- and for ANY sv39 upper-half address, which includes every
    kernel VA, that byte reads 0xFF. Decoded on the new map 0xFF is
    {sentinel=1, ldc0_valid=1, cnt=63}: "untagged LDCs are saturated routine traffic", read off
    a bitstream with no census at all, with the sentinel looking genuine.

    Switch 193 is sound in BOTH directions: on the current bitstream reg 1 is
    {5'b0, store_buf_commit_cnt}, so bit 7 is hard ZERO by construction rather than merely
    usually zero. 194 ({6'b0, store_state}) has the same property.

    Getting this backwards mis-assembles mepc and manufactures a census, so it is decided once,
    first, off the byte whose old encoding cannot forge the sentinel.
    """
    return b193 is not None and ((b193 >> 7) & 1) == 1


def decode_s07_census(v):
    """216 = {1'b1 sentinel, ldc0_valid, untagged_ldc_cnt[5:0]}."""
    if v is None:
        return None
    if not ((v >> 7) & 1):
        return ("NOT PRESENT", f"0x{v:02x}: sentinel bit7 is clear, so this bitstream has no "
                f"untagged-LDC census. NOT a zero count.")
    cnt = v & 0x3F
    return ("census", f"untagged-LDC responses = {cnt}{' (saturated)' if cnt == 63 else ''}, "
            f"ldc0_valid={(v >> 6) & 1}. NOTE: bits[5:0] survive a selftest fire; bit 6 does "
            f"not, because the control sets it.")


def decode_s07_correlate(b193, b194):
    """193/194: does the recorded producer feed the faulting consumer?

    193 = {1'b1, rd_rs1_match, fault_rs1_valid, ldc_rd[4:0]}
    194 = {1'b1, 2'b0,         fault_rs1[4:0]}

    THIS MUST BE READ BEFORE 208. rd_rs1_match is computed in hardware as
    (rd of the last committed LDC == rs1 of the faulting instruction), which is the four-wedge
    invariant checked on silicon instead of assumed. It exists because a ROLLING record is not
    self-validating either: untagged LDC responses are routine (measured -- boot software
    produces them from miss refills), and any unrelated one completing between producer and
    consumer overwrites the record. A clobbered rolling record is indistinguishable from a
    correct one, which is the same failure the one-shot had, one level up.

      match=1 -> 208 is licensed: the record IS the load that fed the faulting instruction
      match=0 -> 208 carries NO verdict, and 194 says which register the consumer wanted
    """
    if b193 is None or b194 is None:
        return None
    if not ((b193 >> 7) & 1) or not ((b194 >> 7) & 1):
        return ("NOT PRESENT", "sentinel bit7 clear on 193/194: this bitstream predates the "
                "producer/consumer correlation. On it, 193 is store_buf_commit_cnt and 194 is "
                "store_state -- do NOT read them as correlation.")
    match = (b193 >> 6) & 1
    rs1_valid = (b193 >> 5) & 1
    ldc_rd = b193 & 0x1F
    fault_rs1 = b194 & 0x1F
    detail = (f"ldc_rd=x{ldc_rd} fault_rs1=x{fault_rs1} rs1_valid={rs1_valid}")
    if match:
        # match=1 CORROBORATES, it does not prove: this is a 5-bit register-number equality, so
        # two unrelated instructions coincide 1 time in 32. Quote it as support, never as proof.
        return ("LICENSED (match=1, corroborates)", detail + " -- the recorded load feeds the "
                "faulting instruction, so 208 is about the right load. NOTE: a 5-bit equality, "
                "so unrelated instructions coincide ~1 in 32; this corroborates rather than "
                "proves.")
    return ("NO VERDICT (match=0)", detail + " -- the recorded load does NOT feed the faulting "
            "instruction, so 208 says nothing about this wedge. The consumer wanted "
            f"x{fault_rs1}; the record is for x{ldc_rd}.")


def decode_s07_verdict(v):
    """Decode the S-07 tag-history verdict byte (switch 208) -> (verdict, detail, fault).

    Layout, verified against capstone-ariane 618f4ce36 core/cva6.sv (bank 3'b110 block opens
    at :1277, this leg at reg 5'b10000), MSB first:

        [7]   ldc0_valid      an LDC came back untagged and was recorded (one-shot)
        [6:5] ldc0_src        0 = L1 hit, 1 = miss refill (tag memory), 2 = write-buffer fwd
        [4]   stc_valid       a capability-granule store was recorded
        [3]   stc_ctag        the tag that store WROTE
        [2]   gran_match      both records refer to the SAME 16-byte granule (computed in HW)
        [1]   stc_clobbered   a PLAIN store later overwrote that granule, so its tag was
                              legitimately cleared and an untagged reload is CORRECT
        [0]   selftest_seen   the sticky was set synthetically, not by a real displacement

    The order of the tests below is the order the RTL lane specified and it matters:
    `clobbered` outranks everything, because a granule whose tag was legitimately cleared by a
    plain store would otherwise read as "stored tagged, came back untagged" -- a confident
    false "hardware tag loss". That bit exists because the sim corpus does exactly that
    sequence in cap-tag-cache, which is how it was found.
    """
    if v is None:
        return None
    ldc_valid = (v >> 7) & 1
    src       = (v >> 5) & 0b11
    stc_valid = (v >> 4) & 1
    stc_ctag  = (v >> 3) & 1
    match     = (v >> 2) & 1
    clobbered = (v >> 1) & 1
    selftest  = v & 1

    # Integrity: fields that are meaningless unless their valid bit is set, and a gran_match
    # that cannot be true without both records. A violation means the READOUT is wrong -- not
    # a finding about the core -- exactly as for the 204 encoding.
    fault = []
    if match and not (ldc_valid and stc_valid):
        fault.append("gran_match set without both records")
    if src and not ldc_valid:
        fault.append("ldc0_src set without ldc0_valid")
    if (stc_ctag or clobbered) and not stc_valid:
        fault.append("stc_ctag/clobbered set without stc_valid")
    if src == 3:
        fault.append("ldc0_src == 3 is not a defined source")

    # src==0 means the tag came off the L1 DATA ARRAY on this response, which INCLUDES a miss
    # that was refilled and then read from the array. It does NOT mean the line was already
    # resident before this instruction: genuine-hit vs replayed-after-refill is not in this bit.
    srcname = {0: "tag from L1 array (incl. replay after refill)",
               1: "miss refill (tag memory)", 2: "write-buffer forward", 3: "UNDEFINED"}[src]
    bits = (f"ldc0_valid={ldc_valid} src={src} ({srcname}) stc_valid={stc_valid} "
            f"stc_ctag={stc_ctag} gran_match={match} clobbered={clobbered} "
            f"selftest_seen={selftest}")

    if fault:
        return ("INSTRUMENT FAULT", bits + "  <== " + "; ".join(fault)
                + ". The readout is wrong, NOT the core.", True)
    # selftest_seen is NOT a synthetic-record marker, and treating it as one DISCARDS REAL
    # EVIDENCE. It is sticky-until-reset and means "the control ran at some point in this boot".
    # Once the records roll, the next real untagged LDC overwrites the synthetic pattern while
    # this bit still reads 1 -- so a genuine finding after a fire would be thrown away as the
    # control. The authoritative test is the PADDR SENTINEL (0x5A at 205/206, 0x5A5A5A at
    # 219/222/223), which travels with the record itself; this bit only says a fire happened.
    if selftest:
        return ("control has fired (records may still be REAL)",
                bits + "  -- selftest_seen is sticky-until-reset and says only that the control "
                "ran at some point, NOT that this record is synthetic. Check the paddr sentinel "
                "(0x5A at 205/206) to decide: sentinel present => synthetic, absent => this is "
                "a real record that overwrote the injected one.", False)
    if clobbered:
        return ("NO VERDICT (clobbered)", bits + "  -- a plain store later overwrote that "
                "granule, so its tag was legitimately cleared and an untagged reload is "
                "CORRECT. Says nothing about tag loss.", False)
    if match and stc_ctag:
        return ("(b) GENUINE TAG LOSS", bits + f"  -- the store WROTE tag=1 and the load read "
                f"0 on the same granule; source says where: {srcname}.", False)
    if match and not stc_ctag:
        return ("(c) STORED UNTAGGED", bits + "  -- the granule was stored with tag=0, so the "
                "reload returning NOT_CAP is CORRECT and the fault is UPSTREAM of both memory "
                "and the syncer.", False)
    # NOTE FOR THE stc_pc APERTURE, because misreading it kills a CORRECT hypothesis.
    #
    # stc_pc is only a test of "which code filled this granule" when gran_match == 1. The store
    # record ROLLS, so with match == 0 it holds merely the most recent capability store before
    # the freeze -- not the store that filled the faulting granule. A workload PC there is then
    # FULLY CONSISTENT with a teardown-triggered wedge: if the close path performs no capability
    # stores after entry, the last STC is legitimately from the workload while the fault is
    # entirely in teardown. Read 193 -> gran_match -> only then treat stc_pc as evidence.
    #
    # Also: stc_pc latches at COMMIT of the STC while the granule record latches at store-buffer
    # PUSH. Same instruction in the ordinary case, but different events -- if they ever disagree,
    # the PADDR is authoritative, because gran_match is computed from it.
    if ldc_valid and not match:
        # match=0 IS WEAK. The reason given here used to be an asymmetry -- rolling STC vs
        # one-shot LDC -- and that was wrong: BOTH records roll (see S07_RECORDS_ROLL). The
        # correct reason is symmetric and no weaker: this compares the most recent untagged load
        # against the most recent capability store, and EITHER may have rolled past the granule
        # of interest since the event under investigation. match=1 is the informative direction;
        # match=0 is equally consistent with either record having moved on. Do NOT report it as
        # "the granule was not the recorded store".
        return ("unmatched (WEAK)", bits + "  -- the records do not refer to the same granule, "
                "but BOTH records roll -- the STC on every capability store, the LDC on every "
                "untagged response -- so this is equally consistent with either having rolled "
                "past. Carries far less weight than a match; compare the granule addresses.",
                False)
    if not ldc_valid:
        return ("no untagged load seen", bits, False)
    return ("unclassified", bits, False)


def decode_s07_cursor(obs):
    """The S-07 H1/H2 verdict read from the pMethods MEMORY SLOT. Returns (verdict, detail).

    Why the slot and not `mtval`: the RTL lane's mtval diagnostic reports the faulting rs1
    cursor, but a capability fault inside a domain WEDGES at exception commit instead of
    trapping to mtvec (capstone-ariane core/cva6.sv:1228-1231), so the monitor's dump never
    runs and mtval is unreadable on this path. Reading the slot is also strictly stronger:
    if the LOAD_WB path erases the capability in the register -- the consequence the RTL lane
    CONFIRMED in sim -- then the register cursor is 0 under H1 too, and mtval would read H2.
    """
    if obs is None:
        return None
    tag = obs >> 24
    if tag == 0x51:
        return ("H1", f"cursor[23:0]=0x{obs & 0xFFFFFF:06x} NON-ZERO -- a real capability "
                      f"arrived NOT_CAP; S-07 is a silicon defect")
    if tag == 0x52:
        return ("H2", f"cursor ZERO (metadata[23:0]=0x{obs & 0xFFFFFF:06x}) -- pMethods really "
                      f"was NULL; this site is a correct null deref, hunt upstream")
    if tag == 0x53:
        return ("no-hit", f"reached sqlite3OsRead {(obs >> 8) & 0xFFFF} time(s), "
                          f"{obs & 0xFF} untagged -- the site never went bad in this run")
    if tag == 0x54:
        return ("PAGER-FIELD", f"pPager->pMmapFreelist was ALREADY non-zero on entry "
                               f"(cursor[23:0]=0x{obs & 0xFFFFFF:06x}) -- the value was wrong "
                               f"before pagerFreeMapHdrs touched it; look upstream")
    if tag == 0x55:
        return ("PAGER-SPILL", f"the field read ZERO but the stack copy of p came back non-zero "
                               f"(cursor[23:0]=0x{obs & 0xFFFFFF:06x}) -- the stc/ldc stack round "
                               f"trip is where it changed")
    if tag == 0x56:
        return ("pager-clean", f"pagerFreeMapHdrs ran {obs & 0xFFFF} time(s), every read all-zero "
                               f"-- the site behaved as the source requires")
    if tag == 0x57:
        ok = obs == 0x57070703
        return ("selftest " + ("PASS" if ok else "FAIL"),
                f"type(nonzero plant)={(obs >> 16) & 0xFF} (expect 7), "
                f"type(zero plant)={(obs >> 8) & 0xFF} (expect 7), "
                f"ld-match flags=0b{obs & 0xFF:02b} (expect 0b11)"
                + ("" if ok else "  <== INSTRUMENT UNPROVEN: any 0x51/0x52 verdict is void"))
    return None


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
    if os.environ.get("FPGA_IMG_NAME"):
        # The staleness guard compares the LOCAL firmware against the LOCAL domains. When booting
        # a stored server-side image neither side of that comparison is what will run, so the check
        # would be answering a question nobody asked. Skipped, and said out loud.
        print("[stages] FPGA_IMG_NAME set: skipping the local firmware/initramfs staleness check "
              "-- the image that boots is the stored one, not the local build.", file=sys.stderr)
    else:
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
        # BOOT AN IMAGE ALREADY ON THE SERVER, without uploading and without a local copy.
        #
        # The console stores boot images under a CONTENT-HASH name (sha256[:12]), so a name can
        # only ever carry the bytes it was uploaded with -- it cannot be overwritten with
        # different content. That makes the store an archive of every firmware ever run, and the
        # only way back to an exact past image once the local build tree has moved on: the build
        # is NOT reproducible (BR2_REPRODUCIBLE is off, so the cpio carries per-file mtimes, and
        # the kernel embeds an incrementing .version plus a build timestamp), so a byte-identical
        # rebuild of a past firmware is not achievable.
        #
        # This exists because a sporadic silicon defect stopped reproducing and the question
        # "does it still happen on the exact image where it did?" is otherwise unanswerable.
        #
        # Set FPGA_IMG_NAME to a stored name to use it. The local-artifact checks are skipped
        # because there is no local artifact to check -- which is stated loudly rather than
        # silently, since it means the usual staleness guard is NOT protecting this run.
        _img_override = os.environ.get("FPGA_IMG_NAME")
        if _img_override:
            log(f"!! BOOTING STORED IMAGE {_img_override!r} FROM THE CONSOLE -- no upload, and "
                f"no local artifact to verify it against.")
            log("!! The initramfs-embeds-current-domains check does NOT apply to this run.")
            _img_name = _img_override
        else:
            console.upload_boot_image(IMG_NAME, str(IMG))
            _img_name = IMG_NAME
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
                cold_boot(console, C.GDB_PROMPT, _img_name)
                break
            except (ActionTimeout, RuntimeError) as _e:
                log(f"  cold_boot failed ({_e}); retry {_boot_attempt}/3")
                if _boot_attempt == 3:
                    raise
        log(f"booted once; running {len(DOMS)} staged domains in sequence")

        # PRE-RUN BASELINE: is the one-shot LDC record already spent before ANY domain runs?
        #
        # Measured 2026-08-18: the record was already latched after the control on the very
        # first boot, with a byte that never changed again all boot. That says the boot carries
        # no S-07 verdict, but NOT who spent it. This read separates the two:
        #   ldc0_valid=1 HERE  -> spent by Linux/OpenSBI/the entry glue before any domain, so
        #                         no arrangement of domains can rescue it and the record needs
        #                         to be clearable in RTL;
        #   ldc0_valid=0 HERE  -> spent by the control domain, so running the workload FIRST
        #                         (accepting a weaker boot-validity argument) would still get a
        #                         usable record.
        # Nothing but reset clears it -- the 191 trap-log clear touches no s07_* register -- so
        # this is the only point in the boot where the answer is visible.
        # Which probe generation is in this bitstream? Decided ONCE, here, because it changes
        # how mepc is assembled: gen 2 reclaims switches 201-203 for stc_pc, so an eight-byte
        # mepc assembly would silently produce a plausible wrong address. Everything this
        # campaign has localised rests on that number, so the discriminator is read before any
        # of it.
        _probe_gen2 = False
        try:
            def _rd(_v):
                for _b in range(8):
                    console.set_switch(_b, bool(_v & (1 << _b)))
                time.sleep(1.2)
                _s = console.latest(C.LISTEN.get("led_state", "led_state"))
                _b_ = _s.get("states") if isinstance(_s, dict) else None
                return sum((1 << i) for i, b in enumerate(_b_) if b) if _b_ else None

            # Generation FIRST, off 193 -- see decode_probe_generation for why not 216.
            _g193 = _rd(193)
            _probe_gen2 = decode_probe_generation(_g193)
            _cl = (f"  [s07] probe generation: "
                   f"{'2 (rolling records, census, correlation)' if _probe_gen2 else '1 (one-shot records)'}"
                   f"   sw=193 = " + ("UNREAD" if _g193 is None else f"0x{_g193:02x}")
                   + "  (193 is the discriminator: on gen 1 it is {5'b0,store_buf_commit_cnt}, "
                     "so bit7 is hard zero and cannot forge the sentinel -- unlike 216, whose "
                     "gen-1 value tval[47:40] reads 0xFF for any upper-half address and would "
                     "decode as a saturated census)")
            print(_cl, flush=True)
            transcript.append(_cl + "\n")

            # The census is only READ once the generation is settled, and only then decoded.
            if _probe_gen2:
                _cen = _rd(216)
                _cd = decode_s07_census(_cen)
                _cl2 = ("  [s07] census: sw=216 = "
                        + ("UNREAD" if _cen is None else f"0x{_cen:02x}")
                        + (f"  {_cd[0]}: {_cd[1]}" if _cd else ""))
                print(_cl2, flush=True)
                transcript.append(_cl2 + "\n")

            _v0 = _rd(208)
            for _b in range(8):
                console.set_switch(_b, False)
            _d0 = decode_s07_verdict(_v0)
            _l0 = ("  [s07] PRE-RUN baseline: sw=208 "
                   + ("UNREAD" if _v0 is None else f"0x{_v0:02x} {_v0:08b}")
                   + (f"  {_d0[0]}: {_d0[1]}" if _d0 else ""))
            print(_l0, flush=True)
            transcript.append(_l0 + "\n")
            if _v0 is not None and ((_v0 >> 7) & 1):
                _l1 = (("  [s07] ROLLING RECORD, NOT SPENT: ldc0_valid=1 before any domain ran "
                        "means boot software produced at least one untagged LDC, which is "
                        "ROUTINE (miss refills over scalar data do it). On this bitstream the "
                        "record OVERWRITES on every untagged response (load_unit.sv:774), so a "
                        "later workload load will replace this one and 208 stays usable all "
                        "boot. Read gran_match with clobbered=0, never ldc0_valid alone.")
                       if S07_RECORDS_ROLL else
                       ("  [s07] SPENT BEFORE ANY DOMAIN RAN -- the LDC one-shot was latched by "
                        "boot-time software (Linux/OpenSBI/glue), not by the workload. No "
                        "ordering of domains can rescue it; the record must become clearable in "
                        "RTL. Everything 208 reports this boot is about that earlier load."))
                print(_l1, flush=True)
                transcript.append(_l1 + "\n")
        except Exception as exc:
            log(f"pre-run 208 baseline failed ({type(exc).__name__})")

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
            #
            # 191 IS ALSO A BLIND WINDOW, and the bias runs the wrong way. The logging always_ff
            # is `if (dom_switch_log_clear) <clear> else <record>`, so while the switches sit at
            # 191 the record branch does not run and ANY displacement in that window is lost --
            # and a lost displacement looks exactly like case (b), "memory did it". Two rules
            # follow and both are respected here:
            #   * never park at 191. The window below is bounded by one sleep and is over before
            #     the domain starts, so the domain's own traffic is fully recorded; only monitor
            #     and kernel traffic during the clear can go unseen, which cannot manufacture a
            #     false case (a).
            #   * never hit 191 after a wedge whose latch has not been read yet -- it would wipe
            #     recent_nontrivial_mcause/mepc/seen. Safe today because the loop breaks
            #     immediately after the wedge reads and never reaches another clear.
            # The S-07 displacement sticky is NOT in the clear list, so that evidence survives a
            # log clear; only the trap latch is affected.
            _do_clear = (TRAPLOG_CLEAR == "all"
                         or (TRAPLOG_CLEAR == "first" and dom_idx == 1))
            if not _do_clear:
                log(f"trap-log clear SKIPPED for {label} (SQLITE_TRAPLOG_CLEAR={TRAPLOG_CLEAR}) "
                    f"-- zero blind window for this domain; trap fields are last-writer-wins, "
                    f"so compare against the pre-run latch below before trusting them")
            else:
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
            # int(...), NOT the re.Match. m_obs is the match object; the int lives in group(1),
            # exactly as line 313 and line 330 already do it. Passing the match here raised
            # TypeError inside decode_probe and killed a boot at the first probe arm.
            _retry = decode_s07_retry(int(m_obs.group(1))) if m_obs is not None else None
            if not wedged and bad and _retry is not None:
                a, b, c = _retry
                _tag = int(m_obs.group(1)) >> 24
                if _tag == 0x5E:
                    log(f"  {dom}: S-07 CAUGHT AT POINT OF USE -- atuse={a} null={b} seen={c}. "
                        f"The capability was TAGGED when queried and NOT_CAP a few instructions "
                        f"later at the deref, with only a spill/reload in between. "
                        + ("Some were NULL (H2)." if b else "None were NULL: a genuine LOST TAG (H1)."))
                    bad = False
                    results.append((dom, False, int(m_obs.group(1)), True, True, True, None)) if False else None
                _null = _tag == 0x5D
                log(f"  {dom}: S-07 RETRY PROBE -- untagged={a} "
                    + (f"NULL-pMethods={b}" if _null else f"still-untagged-on-retry={b}")
                    + f" recovered-on-retry={c}"
                    + ("  => pMethods is genuinely NULL: this site is a correct NULL deref, "
                       "NOT a lost tag. The real defect is UPSTREAM." if _null else "")
                    + ("  => MEMORY LOST THE TAG (A-2 refill path)" if b and not _null else "")
                    + ("  => MEMORY WAS FINE: register delivery" if c and not b else ""))
                bad = False
            _cur = decode_s07_cursor(int(m_obs.group(1))) if m_obs is not None else None
            if not wedged and bad and _cur is not None:
                _v, _d = _cur
                log(f"  {dom}: S-07 CURSOR PROBE -- {_v}: {_d}")
                bad = False
            _probe = decode_probe(int(m_obs.group(1))) if m_obs is not None else None
            if not wedged and bad and _probe is not None:
                a, b, c = _probe
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
            # EXCX AND THE OTHER MONITOR SPIN TAGS BELONG HERE. This regex decides whether a
            # non-returning arm is reported as "the domain wedged" or "the MONITOR wedged", and it
            # listed only two tags. EXCX:0000E002 -- the unconditional `default:` arm of the
            # monitor's handle_exception, i.e. "took a trap it does not handle at all" -- was not
            # among them, so the clearest possible M-mode monitor wedge was classified as a domain
            # entry stall and the run was declared to carry no verdict. It carried the most
            # important verdict we had; see fpga-repros/S08-*.
            #
            # RGNO is included on the same reasoning and it RECLASSIFIES OLD RUNS, deliberately: the
            # arms this project has been recording as "R-16 entry stalls" at SQ: id=5 are actually
            # deterministic monitor region-pool exhaustion (RGNO:0000E00C, RGNN:00000020 = 32
            # regions). They are still excluded from wedge counts -- but as a known monitor limit,
            # not as an unexplained per-image stall.
            montag  = re.search(r"(SPL[AB]|ILLX|EXCX|RCPX|WCPX|SHAX|RGNO|DPIC|DPIX|DRET):([0-9A-Fa-f]{8})", text)
            results.append((label, wedged, obs, returned, created, entered,
                            montag.group(0) if montag else None))

            # S-07 DISPLACEMENT STICKY BIT, READ AFTER EVERY DOMAIN.
            #
            # THE BIT IS BOOT-CUMULATIVE: it is cleared only by reset, so the byte read at a
            # wedge covers everything since power-on -- the k800 control, every earlier domain
            # in the boot, and the monitor's own capability traffic. A non-zero byte at the
            # wedge is therefore NOT by itself attributable to the domain that wedged, and
            # reading it only once would produce exactly the kind of confident mis-attribution
            # this campaign keeps paying for.
            #
            # So sample it after EVERY domain. The COUNT DELTA across a domain is the
            # attributable number; the seen-bits alone are not on a multi-domain boot.
            #
            # Pre-registered readings, so neither can be rationalised afterwards:
            #   baseline (after the control) NON-ZERO -> displacement happens during ordinary
            #     operation, before the workload under test. That is a bigger finding than S-07
            #     and changes what we chase, so it is reported even when zero.
            #   baseline zero, delta non-zero over the wedging domain -> the producing load's
            #     response was displaced onto a scalar writeback port: the value was intact in
            #     memory (case a). The count says one-off or routine.
            try:
                def _read_sw(_sw):
                    """Read one debug-mux aperture, defending against the LED PULSE STRETCHER.

                    THE LEDS ARE NOT A SNAPSHOT. `ariane_xilinx.sv:956-979` holds each LED bit
                    HIGH for 2^20 cycles (~21 ms at 50 MHz) after the last cycle that bit was
                    driven. So a naive read returns the bitwise OR of EVERY aperture displayed
                    in the preceding window -- and the switch walk visits several on the way.

                    That is exactly what produced the impossible readings on
                    caplifive_s07debug_18august.bit: `src=3`, which is not a defined value at
                    all (it is src=1 from one visited aperture ORed with src=2 from another),
                    and `count=12 with ldc_seen clear`, which the closed encoding is designed to
                    make unrepresentable. One of those same readings decoded cleanly as
                    "genuine tag loss" -- the answer the whole investigation wants -- and would
                    have been published without the encoding's self-check.

                    The stretcher can only turn bits ON, never off: a bit reads 1 if it was high
                    ANYWHERE in the window, and 0 only if it was high nowhere. So every 0x00
                    reading ever taken remains trustworthy (contamination cannot manufacture a
                    zero), and the non-zero readings are the suspect class.

                    TWO defences, because a settle delay alone is not sufficient:

                      1. Wait out the stretcher, several times over.
                      2. Require TWO samples, a full window apart, to AGREE -- and take a FRESH
                         `led_state` event for each. `console.latest()` returns the last payload
                         RECEIVED, so with a cached read "sample twice and compare" compares one
                         stale sample against itself and passes while proving nothing. That is
                         the same shape as a check that cannot fire.

                    Disagreement returns None (VOID). A contaminated aperture must never be
                    handed back as a value."""
                    set_switch_value(console, _sw)

                    def _sample():
                        _mark = console.now()
                        try:
                            _s = console.wait_event(
                                C.LISTEN.get("led_state", "led_state"),
                                timeout=LED_FRESH_TIMEOUT_S, since=_mark)
                        except Exception:
                            # No change since the mark means the value already settled; the
                            # cached payload IS that settled value.
                            _s = console.latest(C.LISTEN.get("led_state", "led_state"))
                        _b = _s.get("states") if isinstance(_s, dict) else None
                        return sum((1 << i) for i, b in enumerate(_b) if b) if _b else None

                    time.sleep(LED_SETTLE_S)
                    _a = _sample()
                    time.sleep(LED_SETTLE_S)
                    _c = _sample()
                    if _a != _c:
                        _m = (f"  [s07] sw={_sw}: VOID -- two reads {LED_SETTLE_S}s apart "
                              f"disagree ({_a} vs {_c}). LED pulse-stretcher contamination, "
                              f"not data.")
                        print(_m, flush=True)
                        transcript.append(_m + "\n")
                        return None
                    return _a

                # UART-HOSTILE SWITCH VALUES -- the low three switch bits are NOT mux selectors.
                #
                #   sw[0]  ariane_xilinx.sv  `uart_debug_takeover = sw[0] | uart_debug_active`
                #          and cva6.sv:913   `uart_debug_tx_o = switches_i[0] ? tracer_uart_tx
                #                             : uart_debug_tx`
                #          -- takes the CONSOLE TX PIN away from the APB UART and gives it to the
                #             tracer. The shell's output is not lost, it is not transmitted.
                #   sw[1]  cva6.sv:941 `dump_enable_i(switches_i[1])` -- arms a ONE-SHOT binary
                #          dump of the trace buffer over that same pin.
                #   sw[2]  cva6.sv:942 `overwrite_i(switches_i[2])` -- tracer ring-buffer mode,
                #          harmless to the console.
                #
                # So a value is UART-safe only if (v & 0b11) == 0. 204 (0xCC) is safe. 255 is
                # NOT: it sets all three, hijacking the console AND arming a dump. A boot was
                # lost to exactly this -- the switches were left at 255, the next domain's shell
                # line was never echoed, and it read as a wedge when the core was fine.
                #
                # A domain that goes quiet because its TX pin was taken is indistinguishable from
                # one that wedged, so the trap summary is sampled ONCE, after the control, purely
                # to give the staleness comparison a pre-test baseline -- never between the
                # domains under test.
                assert (204 & 0b11) == 0, "switch 204 must be UART-safe"
                assert (208 & 0b11) == 0, "switch 208 must be UART-safe"
                _v = _read_sw(204)

                # The tag-history verdict byte. 208 is even, so it is safe to sample between
                # domains alongside 204; it is the byte that separates (b) a genuine tag loss
                # from (c) a granule that was stored untagged, which no software probe can do.
                _w = _read_sw(208)
                _d = decode_s07_verdict(_w)
                _wline = (f"  [s07] after {label}: sw=208 verdict "
                          + ("UNREAD" if _w is None else f"0x{_w:02x} {_w:08b}"))
                if _d is not None:
                    _wline += f"  {_d[0]}: {_d[1]}"
                print(_wline, flush=True)
                transcript.append(_wline + "\n")

                # IS THE ONE-SHOT ALREADY SPENT? This is the control that decides whether the
                # later wedge readout means anything, and it costs one read.
                #
                # CORRECTED 2026-08-18: the LDC record captures the MOST RECENT response
                # returning tag=0, not the first since reset (load_unit.sv:774 has no
                # !s07_ldc0_valid_q guard). The paragraph below described the withdrawn one-shot
                # design and is kept only for the S07_RECORDS_ROLL=0 path. Post-S-06 an untagged LDC
                # is architecturally legal and does not fault; it only faults when the result is
                # later USED as a capability. So any earlier benign untagged load anywhere in
                # the boot -- monitor, entry glue, the control domain, a 128-bit granule copy
                # over scalar data -- latches the record, and the faulting load then never gets
                # recorded. The failure is SILENT: ldc0_valid=1 with a paddr that is not the
                # faulting address and gran_match=0, which is indistinguishable from a genuine
                # unmatched result.
                #
                # Nothing clears it but reset -- the 191 trap-log clear does not touch any s07_*
                # register -- so a spent probe stays spent for the whole boot.
                if _w is not None and dom_idx == 1 and ((_w >> 7) & 1):
                    _sline = (("  [s07] ldc0_valid set after the control -- EXPECTED and harmless "
                               "on a rolling record: it says an untagged LDC has happened, not "
                               "that the probe is used up. Later domains overwrite it. The real "
                               "limit on this bitstream is that gen 1 has no 193/194 correlation "
                               "gate, so a granule match is SUGGESTIVE, not licensed.")
                              if S07_RECORDS_ROLL else
                              ("  [s07] PROBE ALREADY SPENT after the control: ldc0_valid is set "
                               "before any test domain ran, so the LDC record belongs to an "
                               "earlier benign untagged load and CANNOT be the faulting one. "
                               "Any later wedge verdict from 208 carries NO weight this boot."))
                    print(_sline, flush=True)
                    transcript.append(_sline + "\n")
                # The trap-log summary {seen, mcause[6:0]} alongside it. With the clear skipped
                # the trap fields are last-writer-wins, so the ONLY way to tell "this domain
                # trapped" from "a previous domain's trap is still latched" is to compare against
                # the value standing before this domain ran. Sampling it per domain is what makes
                # that comparison possible at all; without it a stale latch reads as a result.
                # THE TRAP SUMMARY IS NOT SAMPLED MID-RUN AT ALL, and it cannot be made safe.
                #
                # It lives only at reg 5'b11111 in bank 3'b111 (cva6.sv:1229), so its switch
                # value is 0b111_11111 = 255 -- inherently ODD, hence always a console hijack.
                # There is no even aperture: 254 is reg 5'b11110, which is
                # rev_node_serving_idx[31:24] (cva6.sv:1228), a different field entirely.
                #
                # Worse than a momentary hijack: the trap-log clear at 191 ARMS a one-shot trace
                # dump (sw[1]), and that dump is edge-triggered and streams the whole buffer, so
                # it outlives the switch value that armed it. Holding any odd value later
                # reconnects the tracer to the console MID-STREAM and injects binary trace bytes
                # into it. With the clear running before the control, a 255 read after the
                # control is exactly that pattern -- it would spray garbage into the console
                # immediately before the domains under test.
                #
                # Dropped rather than worked around, because it is a nicety and the measurement
                # does not need it: the displacement byte does not depend on the trap latch, and
                # a genuine capability fault is self-evident at the wedge anyway (mcause 25 at a
                # DOMAIN VA, versus the kernel VAs and ordinary causes a stale latch shows). The
                # wedge readout still reports the trap summary, where the run is already over and
                # an injected burst costs nothing.
                # Park the switches back at 0 between domains, which is where every earlier
                # version of this runner left them. Sampling leaves them at the last value read,
                # so without this the next domain runs with the mux parked somewhere it never
                # used to be. Only 191 is known to have a side effect (the log clear) and 0 is
                # not it -- but a boot hung between domains once with the switches parked at
                # 255, and restoring the prior resting state removes that as a variable instead
                # of leaving it to be argued about later.
                for bit in range(8):
                    console.set_switch(bit, False)
                if _v is None:
                    _line = f"  [s07] after {label}: sw=204 displacement UNREAD"
                else:
                    _stc, _ldc, _cnt = (_v >> 7) & 1, (_v >> 6) & 1, _v & 0x3F
                    # FREE INTEGRITY CHECK ON THE READOUT PATH, not a finding about the core.
                    #
                    # seen and count move in the SAME branch, count saturates at 63 rather than
                    # wrapping, and nothing clears count except reset. So the encoding is closed
                    # and only three shapes are reachable:
                    #     0x00                 quiescent
                    #     0x80                 stc only -- legal, the STC arm does not touch the
                    #                          LDC counter
                    #     ldc_seen=1 & cnt>=1  0x41-0x7F and 0xC1-0xFF
                    # Everything else is impossible for the design to produce, INCLUDING 0x40 and
                    # 0xC0: the first displacement sets seen and increments in one go, so
                    # ldc_seen with a zero count cannot occur.
                    #
                    # Anything outside the legal set means the READOUT is wrong -- wrong switch, a
                    # garbled byte, or a sampling race -- and the value must not be reported as
                    # evidence of anything. This matters because the silent failure is
                    # indistinguishable from a result: a mis-aimed read returns the mux default
                    # 0x00, which is ALSO the legal quiescent value, so a wrongly-pointed probe
                    # looks exactly like a clean "memory did it" verdict. A count with no
                    # seen-bit is the only pattern that betrays it.
                    _fault = not (_v == 0x00 or _v == 0x80 or (_ldc == 1 and _cnt >= 1))
                    _line = (f"  [s07] after {label}: sw=204 displacement "
                             f"0x{_v:02x} {_v:08b}  seen={{stc:{_stc},ldc:{_ldc}}} count={_cnt}"
                             + ("  <== INSTRUMENT FAULT: count>0 with ldc_seen clear cannot be "
                                "produced by the design (legal: 0x00, 0x80, or ldc_seen with "
                                "count>=1); the readout is wrong, NOT the core. Do not treat "
                                "this byte as data." if _fault else ""))
                print(_line, flush=True)
                transcript.append(_line + "\n")
            except Exception as exc:
                # Never let the extra read cost the run: the domain verdict above is already
                # recorded, and a failed sample is reported rather than swallowed.
                _line = f"  [s07] after {label}: sw=204 read FAILED ({type(exc).__name__})"
                print(_line, flush=True)
                transcript.append(_line + "\n")

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
                                            # S-07 DISPLACEMENT STICKY BIT (RTL lane, 2026-08-16).
                                            # Bank 3'b110 reg 5'b01100, i.e. switch 192+12 = 204,
                                            # the same base the LATCHED mepc bytes below use
                                            # (196 == reg 4). Byte is
                                            # {stc_seen, ldc_seen, ldc_count[5:0]}, count
                                            # saturating at 63, cleared only by reset -- so it
                                            # survives the wedge exactly like the trap latch.
                                            #
                                            # THIS IS THE S-07 DISCRIMINATOR, and it is the only
                                            # channel that reports at every site: mtval is
                                            # written but unreadable here (the monitor's trap
                                            # dump never runs on a capability fault inside a
                                            # domain, and GDB reads CSRs already clobbered by a
                                            # nested trap).
                                            #   non-zero -> a capability op's response was
                                            #     displaced onto a scalar writeback port, which
                                            #     zeroes cap_result at writeback: the value was
                                            #     still INTACT IN MEMORY (case a).
                                            #   all-zero -> nothing was displaced, so the
                                            #     NOT_CAP came from memory/tag state (case b).
                                            # Report the RAW byte; the reading above is an
                                            # interpretation and belongs in the write-up, not in
                                            # the transcript.
                                            (204, "S-07 displacement {stc,ldc,count[5:0]}", "raw"),
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
                        # STALENESS, decided here rather than per-domain: with the clear skipped
                        # the trap latch is last-writer-wins, so a value identical to the
                        # pre-test baseline means no non-trivial trap was latched since, and the
                        # trap fields carry no verdict about the domain that just wedged. The
                        # displacement byte is unaffected either way.
                        if sw == 255 and v is not None and TRAPLOG_CLEAR != "all":
                            # No pre-test baseline is taken any more (reading 255 mid-run injects
                            # trace bytes into the console -- see the sampling block). So
                            # staleness is judged on the value itself: capability causes are
                            # built as 24 + exception_code[3:0] (cva6.sv, the CAP_WB cause
                            # build), i.e. 24..39 -- which covers UNEXPECTED_OPERAND_TYPE 25,
                            # INVALID_CAPABLITY 26 and CAPABLITY_OUT_OF_BOUND 29
                            # (riscv_pkg.sv:349-353). Anything outside that is ordinary kernel
                            # traffic latched since the last clear and says nothing about this
                            # domain. The range is NOT 24..28: that would dismiss a genuine
                            # mcause-29 as if it were kernel noise.
                            _c = v & 0x7F
                            if not (24 <= _c <= 39):
                                line += (f"   <== cause {_c} is not a capability fault (24..39); "
                                         f"with the clear skipped this is ordinary traffic "
                                         f"latched earlier and says NOTHING about this domain")
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
                    mepc, _mnote = assemble_mepc(mepc_bytes, _probe_gen2)
                    if mepc is not None:
                        print(f"  [wedge] mepc assembly: {_mnote}", flush=True)
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
                        print(f"  [wedge] trap mepc {_mnote}", flush=True)

                    # THE ARCHITECTURAL mtval, VIA GDB -- the only channel that reports the
                    # faulting OPERAND at ANY site.
                    #
                    # caplifive_s07diag.bit puts the faulting rs1 cursor in mtval, but a
                    # capability fault inside a domain WEDGES at exception commit instead of
                    # trapping to mtvec (capstone-ariane core/cva6.sv:1228-1231), so the
                    # monitor's EXCX/MCAU/MEPC/MTVL dump never runs -- measured, six mcause-25
                    # transcripts with no dump while mcause-8 wedges in the same capture print
                    # one. The debug mux carries no tval either. Halting the core and reading
                    # the CSR is what is left.
                    #
                    # IT IS ONLY BELIEVED IF IT AGREES WITH THE LATCH. The core is wedged, the
                    # debug register path has twice returned AXI error-slave junk, and a nested
                    # trap would overwrite the CSRs with a second, unrelated trap. So mtval is
                    # accepted only when gdb's own mepc/mcause match the latched pair -- that is
                    # a real positive control, not a formality: it fails loudly rather than
                    # handing back a plausible wrong operand.
                    if os.environ.get("WEDGE_GDB_MTVAL", "1") == "1":
                        # Same generation-aware assembly as above -- an 8-byte assembly on a
                        # gen-2 bitstream would compare gdb's mepc against a garbage latch and
                        # discard a perfectly good reading (or, worse, accept a wrong one).
                        _latched, _ = assemble_mepc(mepc_bytes, _probe_gen2)
                        try:
                            console.gdb_start()
                            console.gdb_cmd("monitor halt", C.GDB_PROMPT, timeout=30.0)
                            _csr = {}
                            for _e in ("$mcause", "$mepc", "$mtval"):
                                _s = len(console.gdb_text)
                                console._emit("gdb_input", text=f"p/x {_e}\n")
                                try:
                                    _m = console.wait_gdb(r"\$\d+ = 0x[0-9a-fA-F]+",
                                                          timeout=25.0, search_from=_s)
                                    _csr[_e] = int(_m.group(0).split("=")[1].strip(), 16)
                                except Exception:
                                    _csr[_e] = None
                            print(f"  [wedge] gdb CSRs: mcause={_csr['$mcause']} "
                                  f"mepc={_csr['$mepc']} mtval={_csr['$mtval']}", flush=True)

                            # HALTED RE-READ OF THE MUX -- the control this instrument has
                            # never had, and it costs nothing because the hart is ALREADY
                            # halted here for the CSR reads above.
                            #
                            # `debug_led` has exactly one driver, the cva6 mux, but its inputs
                            # include per-cycle signals (commit pc bytes, GPR/vaddr bytes). Each
                            # LED bit is then held high for 2^20 cycles (~21 ms) by the pulse
                            # stretcher in ariane_xilinx.sv:956-979. A WEDGED core is not idle --
                            # it spins in M-mode, committing continuously -- so those inputs keep
                            # toggling, the stretcher keeps reloading, and the byte saturates.
                            # That is why a 0.5 s settle and a two-sample agreement check BOTH
                            # passed while returning 0xfe: the contamination is persistent, not a
                            # decaying tail, and two samples of a saturated value agree perfectly.
                            #
                            # Halting stops the commits, so if the mechanism is activity-driven
                            # reload the halted read is clean and the running read is not. Same
                            # aperture, same boot, one variable. If both are 0xfe the mechanism
                            # is not activity and the readout path itself is broken.
                            #
                            # Reported side by side rather than replacing the earlier value: a
                            # disagreement is the RESULT here, not an error to be smoothed over.
                            try:
                                for _ap in (204, 208):
                                    _halted = _read_sw(_ap)
                                    print(f"  [wedge] HALTED re-read sw={_ap}: "
                                          + ("VOID (samples disagreed)" if _halted is None
                                             else f"0x{_halted:02x} {_halted:08b}")
                                          + "   <-- compare against the RUNNING read above; "
                                            "clean here + saturated there = activity-driven "
                                            "reload, both saturated = readout path",
                                          flush=True)
                            except Exception as _exc:
                                print(f"  [wedge] halted mux re-read failed "
                                      f"({type(_exc).__name__}) -- no verdict from it",
                                      flush=True)
                            _junk = (0xca11ab1ebadcab1e, None)
                            if _csr["$mtval"] in _junk:
                                print("  [wedge] mtval NOT READ (junk/unreadable) -- no operand",
                                      flush=True)
                            elif _csr["$mcause"] != 25 or _csr["$mepc"] != _latched:
                                print(f"  [wedge] mtval DISCARDED: gdb mcause/mepc "
                                      f"({_csr['$mcause']}/{_csr['$mepc']}) do not match the "
                                      f"latched trap (25/{_latched}). The CSRs were clobbered by "
                                      f"a later trap, so mtval belongs to a DIFFERENT fault.",
                                      flush=True)
                            else:
                                print(f"  [wedge] FAULTING OPERAND CURSOR = "
                                      f"0x{_csr['$mtval']:016x}  "
                                      + ("(ZERO -> the operand really was NULL/integer)"
                                         if _csr["$mtval"] == 0 else
                                         "(NON-ZERO -> a non-null value arrived NOT_CAP)"),
                                      flush=True)
                        except Exception as exc:
                            print(f"  [wedge] gdb mtval read failed ({type(exc).__name__}) -- "
                                  f"continuing to release the board", flush=True)
                        finally:
                            # MANDATORY. release_board() does NOT stop gdb (safe_cleanup.py:75-86
                            # does switches/power/unlock only), and an orphaned server-side GDB
                            # session survives a power cycle: every later run then times out
                            # before load_image, and only gdb_stop() clears it. Leaving this out
                            # would let one wedge poison every subsequent board session.
                            try:
                                console.gdb_stop()
                            except Exception as exc2:
                                print(f"  [wedge] gdb_stop FAILED ({type(exc2).__name__}) -- the "
                                      f"session may be orphaned; if the next run times out before "
                                      f"load_image, clear it with --stop", flush=True)
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

        # ---------------------------------------------------------------------------------
        # SELFTEST: prove the displacement detector can fire, on THIS silicon, THIS boot.
        #
        # Runs last, after every domain and after any wedge readout, and that ordering is
        # deliberate: the trigger sets switch 204's pair (ldc_seen and the count) as well as
        # bit 0 of 208, so firing it earlier would contaminate the very byte being measured.
        # It does NOT touch the 208 verdict fields (ldc0_*, stc_*, gran_match), so those stay
        # trustworthy either way.
        #
        # SAFE AFTER A WEDGE: the trigger is combinational off the switch value and the capture
        # sits in an always_ff whose only enables are clock, reset and the trap-log clear --
        # nothing in the fetch/issue/commit/memory path. A wedged core still has a running
        # clock, so this validates a wedging boot exactly as well as a clean one. Only a reset
        # voids it.
        #
        # THE TRAP: switch 220 is BOTH trigger and readback -- reading it arms it. There is no
        # way to ask "is it armed yet" without arming it, so 204 read BEFORE the trigger is the
        # only usable baseline.
        #
        # Without this, every 0x00 is an argued negative: "no displacement happened" and "the
        # detector does not work in this bitstream" are otherwise indistinguishable.
        try:
            def _sw(_v):
                for _b in range(8):
                    console.set_switch(_b, bool(_v & (1 << _b)))
                time.sleep(1.2)
                _s = console.latest(C.LISTEN.get("led_state", "led_state"))
                _bits = _s.get("states") if isinstance(_s, dict) else None
                return sum((1 << i) for i, b in enumerate(_bits) if b) if _bits else None

            _pre = _sw(204)                      # baseline BEFORE arming
            _flag = _sw(220)                     # selecting 220 IS the trigger
            _post = _sw(204)
            _verd = _sw(208)
            for _b in range(8):                  # park somewhere even and harmless
                console.set_switch(_b, False)

            _ok_flag = _flag == 0x01
            _ok_move = (_pre is not None and _post is not None
                        and ((_post >> 6) & 1) == 1
                        and (_post & 0x3F) == min(63, (_pre & 0x3F) + 1))
            _ok_mark = _verd is not None and (_verd & 1) == 1
            for _t in (f"  [s07] SELFTEST pre-204  = "
                       + ("UNREAD" if _pre is None else f"0x{_pre:02x}"),
                       f"  [s07] SELFTEST 220 flag = "
                       + ("UNREAD" if _flag is None else f"0x{_flag:02x}")
                       + ("  (expect 0x01)" if not _ok_flag else "  OK"),
                       f"  [s07] SELFTEST post-204 = "
                       + ("UNREAD" if _post is None else f"0x{_post:02x}")
                       + ("  OK: ldc_seen set and count moved by exactly 1" if _ok_move
                          else "  <== DID NOT MOVE AS EXPECTED"),
                       f"  [s07] SELFTEST 208 bit0 = "
                       + ("UNREAD" if _verd is None else f"0x{_verd:02x}")
                       + ("  OK: set marked SYNTHETIC" if _ok_mark else "  <== NOT MARKED"),
                       ("  [s07] SELFTEST PASS -- the detector fires on this silicon, so every "
                        "0x00 read this boot is a CONTROLLED negative"
                        if (_ok_flag and _ok_move and _ok_mark) else
                        "  [s07] SELFTEST FAIL -- the detector was NOT shown to fire, so every "
                        "0x00 read this boot is an ARGUED negative and carries no verdict")):
                print(_t, flush=True)
                transcript.append(_t + "\n")
        except Exception as exc:
            _t = (f"  [s07] SELFTEST could not be run ({type(exc).__name__}) -- treat this "
                  f"boot's zeros as argued, not controlled")
            print(_t, flush=True)
            transcript.append(_t + "\n")

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
            elif decode_s07_cursor(obs):
                _v, _d = decode_s07_cursor(obs)
                verdict = f"S-07 CURSOR PROBE -- {_v}: {_d}"
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
