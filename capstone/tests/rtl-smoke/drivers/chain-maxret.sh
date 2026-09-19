#!/bin/bash
# chain-maxret.sh -- THREE boots that separate the two things M1_MAXRET moves at once, and bank the two
# Design arms still unmeasured. One image per boot, ascending in risk, each with its own emulator pass.
#
# WHY. Boot 1 of the M1 series wedged at domain ENTRY with mcause 28 = RISCV_EXCP_CAP_OOB and mepc at
# _start+0xec -- the entry glue's byte-wise zeroing store, the instruction being byte-identical in the
# image that works and the image that wedges. So the fault is a capability bound, not pool exhaustion
# (cause 30) and not a failed kernel allocation (which fails domain creation before anything runs).
#
# M1_MAXRET moves TWO variables together, which is why boot 1 could not attribute its own wedge:
#   1. the STORAGE CARVE the entry glue walks at startup -- one descending bump-allocated block per
#      global, 76,832 B at M1_MAXRET=4096 and 1,059,808 B at 65,532;
#   2. the ALLOCATION ORDER of the region the module hands the domain. The model, verified against two
#      built images, is declared = 278,064 + 16*M1_MAXRET, rounded with code_len + 8,192 up to a
#      power-of-two page count: 4,096 lands at order 7, 65,532 at order 9, and ORDER 8 WAS NEVER BUILT.
#
# The matched pair, differing by exactly one thing each (CLAUDE.md: pair a failing arm with a passing
# arm that differs by ONE thing):
#
#   boot 1  249cfda958f22f16  M1_MAXRET=43296  storage 704,032  ORDER 8   <- the untested order
#   boot 2  29d9099326304ee5  M1_MAXRET=43296  storage 704,048  ORDER 8   <- boot 1's image plus ONE
#                                                                           define: phase 2 at
#                                                                           buffer-full instead of 10C
#   boot 3  c685b0d7a95492ef  M1_MAXRET=4096   storage  76,832  ORDER 9   <- the PILOT's carve, byte for
#                                                                           byte, reaching order 9 by
#                                                                           padding R1_STACK to 1 MiB
#                                                                           ALONE. Nothing else differs
#                                                                           from 1b7a04fe237e1580.
#
# BOOTS 1 AND 2 ARE THE LEAD'S RULING OF 2026-09-18 that both release variants be run: boot 1 keeps the
# approved algorithm and reduces C until phase 2 is reachable (C = 4,329, 6.6 % of production capacity),
# boot 2 keeps C at the production capacity and moves the trigger. Boot 2's transcript is NOT the
# protocol's release arm and must never be reported as one; its start line carries rel_at_buffer=1 and a
# lowered target so it says so itself.
#
# PREDICTED READINGS, written down before the boots because an uninformative prediction is a no-go:
#   boot 3 WEDGES    -> the region SIZE is the trigger; the storage carve is innocent and M1_MAXRET is
#                       innocent by association. The pressure arm's ceiling is then whatever keeps the
#                       order at 8, i.e. 43,296, and boot 1 must have passed.
#   boot 2           -> the production-capacity release arm under a moved trigger; only meaningful if boot 1
#                       completed, since it is the same allocation order, and it is SKIPPED if boot 1 did not.
#   boot 3 COMPLETES -> the region size is fine at order 9; the trigger is the storage carve total, and
#                       boot 1's wedge is about a megabyte of globals rather than about the region.
#   boot 1 COMPLETES -> banks BOTH remaining Design arms: pressure holding 43,296 stale references (NOT 66 % of the pool -- see the retraction in the
#                       2026-09-19 bundle; coverage is set by M1_LIVE, not by this buffer), and release at C = 4,329 reaching phase 2 for the first time on
#                       silicon (10C = 43,290 <= 43,296, checked against the loop).
#   boot 1 WEDGES    -> the ceiling is below 704 KB of storage; bisect between 76,832 and 704,032 next.
# Every combination is informative, which is what makes the pair worth two boots.
#
#   MAXRET_HOST=<sqlite_host_rr.user> bash chain-maxret.sh
#
# Both images are linked at DOMAIN_BASE_VA=0x410000 and carry their own emulator pass record. Note that
# the emulator CANNOT reproduce either question: it runs the arms at C = 64, where the buffer never
# binds, and it does not model the module's buddy allocation at all.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
U=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/unify; mkdir -p "$U"
L=$R/capstone/tests/rtl-smoke/drivers/lists
A=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/m1-repro
: "${MAXRET_HOST:?set MAXRET_HOST=<path to sqlite_host_rr.user, the readback host>}"
export FPGA_BITSTREAM=${FPGA_BITSTREAM:-caplifive_m1_054cea69b.bit}

COMMON="Start gate 3 of 3; completion frame (b), M1 MEASURED and condition 4 ANSWERED rather than satisfied. Boot 1 of the series wedged at domain ENTRY with mcause 28 = RISCV_EXCP_CAP_OOB and mepc at _start+0xec, the entry glue's byte-wise zeroing store, an instruction byte-identical in the working and the wedging image -- so a capability BOUND, not exhaustion (cause 30) and not a failed kernel allocation. M1_MAXRET moves two variables at once, the storage carve the entry glue walks and the allocation order of the region the module hands the domain, and these two boots separate them. CONTROLS: k800 retval=4 first and last; the start line's maxret must equal the built image, which is also how the image is identified since this bitstream exposes no digest for a resident image; monitor 2dcd3a5; firmware scanned rc=0 before the boot. EXPECT RETIREMENT AND DO NOT LOG IT AS A LEAK DEFECT: an index retires permanently after 16,384 allocations of itself. The pool does not reset between invocations (R-12); boot 2 of the series consumed 1,314,737 allocations in one boot without exhausting 65,532 indices, which bounds total index consumption at c < 0.0498."

run() { # tag desc img hash list qemulog prereg
  R1_BOOT=1 R1_OUT_TAG=$1 R1_BOOT_TAG=sw8x-mx R1_BOOT_DESC="$2" R1_PREREG="$7" \
  R1_IMG=$3 R1_HASH=$4 R1_HOST=$MAXRET_HOST R1_LIST=$5 \
  R1_QEMU_LOG=$6 R1_QEMU_GATE='R1 m1 end' R1_QEMU_GATE_MIN=1 \
    bash "$R/capstone/tests/rtl-smoke/drivers/board-r1e4.sh" > "$U/board-$1.console" 2>&1
  echo "MAXRET_${1}_EXIT=$?" >> "$U/board-$1.console"
  echo "$(date +%T) $1: $(tail -1 "$U/board-$1.console")"
}

cd "$R"

run mx43296 \
  "the largest retained-reference buffer that stays at allocation order 8: the retain-pressure arm at 66.1 percent of the pool, and the first release arm on silicon to reach phase 2" \
  "$A/maxret43296/r1_slots_pools.dom" 249cfda958f22f16 "$L/m1-maxret43296.txt" \
  "$A/maxret43296/qemu-pressure/boot.log" \
  "PRIMARY -- the two Design arms still unmeasured. PRESSURE always stops at alloc = M1_MAXRET + 1 whatever C is, so its result is the number of stale references HELD and NOT a fraction of the pool -- RETRACTED 2026-09-19: retained references are not distinct indices, roughly M1_LIVE=16 of them recycle however large the buffer, and one boot's 1,314,737 allocations against a 65,532 pool without exhaustion proves it; a clean null is the EXPECTED and CORRECT result, because no old reference is consulted at reclaim and safety is carried by the generation, so the arm is interesting only if it FAILS. RELEASE runs at C = 4,329 so that 10C = 43,290 <= 43,296 and phase 2 is reachable for the first time on silicon; that is 6.6 percent of the protocol's production capacity and the reduction is stated rather than hidden. SECONDARY -- cost: take_cyc/n and give_cyc/n flat, as boot 2 found at 72.03 sd 0.013 and 103.06 sd 0.029; a rise is the finding. ALSO READ: whether the give-cost step boot 2 saw between its first and second snapshot in the arms that retain an alias reappears here -- it is observed and unexplained, and the hypothesis on record is that the snapshot's own capability type read through a stale alias perturbs the trackers. THIS BOOT ALSO TESTS ALLOCATION ORDER 8, WHICH HAS NEVER BEEN BUILT: order 7 works and order 9 wedged. COMPLETES means order 8 is fine and both arms are banked; WEDGES means the ceiling is below 704 KB of storage and the next step is a bisection between 76,832 and 704,128, not another arm. $COMMON"

# Boot 2 is the same allocation order as boot 1, so if boot 1 did not produce an arm there is nothing
# here to learn and the boot is skipped rather than spent. A skip is printed, never silent.
#
# CHECK THE RESULTS FILE, NOT THE CONSOLE. The first version of this guard grepped the .console for
# "R1 m1 end", which that file NEVER contains -- the driver writes arm lines to board-<tag>-b1/ and the
# console carries only its own classification, "R1 end lines: 2". So the condition was unsatisfiable and
# the boot was skipped on 2026-09-19 with the message "boot 1 produced no arm end line" while boot 1 had
# in fact produced two. A guard that cannot pass is not a conservative guard, it is a broken one, and it
# fails in the direction that looks like caution.
if grep -aq 'R1 m1 end' "$U/board-mx43296-b1/r1-lines.txt" 2>/dev/null; then
run mxrelbuf \
  "the release arm at the FULL production capacity under a re-specified phase-2 trigger: buffer-full instead of 10C" \
  "$A/relbuf43296/r1_slots_pools.dom" 29d9099326304ee5 "$L/m1-relbuf43296.txt" \
  "$A/relbuf43296/qemu-release-trig/boot.log" \
  "PRIMARY -- the release arm at C = 65,532, the protocol's production capacity, which the approved algorithm cannot reach: it retains one 16-byte capability per allocation until phase 2, so 10C needs 10,485,120 bytes against a ~2 MB region. This image is boot 1's image plus ONE define, M1_RELEASE_AT_BUFFER=1, which starts phase 2 when the retained buffer fills. THIS IS NOT THE PROTOCOL'S RELEASE ARM and its transcript must never be reported as one; the START LINE says so itself, carrying rel_at_buffer=1 and a target lowered to 43,295 rather than 655,320, which is the field to read and the only one that changes. READ: phase 2 must be ENTERED -- an R1 m1 released line at alloc = 43,295 -- and the arm must then run 2C = 131,064 further allocations with every retained reference cleared, reaching stop=target. The retained set at the moment of release covers 43,295/65,532 = 66.06 percent of the pool, and THAT FRACTION IS THE RESULT, not the allocation count. A clean run is the expected outcome for the same reason the pressure arm's null is expected: no old reference is consulted at reclaim. THE DEFINE IS OPT-IN AND VERIFIED FREE: with it present but off, M1_MAXRET=4096 rebuilds to 1b7a04fe237e1580 byte for byte, the image currently on the board.  EMULATOR EVIDENCE, TWO-SIDED, on record before this boot: at --cap 64 this image behaves identically to the unmodified one (released at 640 = 10C, stop=target, only the start-line annotation differs), and at --cap 4400 the trigger fires -- target lowered to 43,295, released at 43,295, stop=target at 52,095 = 43,295 + 2C. A pass at a capacity where the buffer cannot fill would have proved nothing about the change, which is why both were run. $COMMON"
else
  echo "$(date +%T) mxrelbuf: SKIPPED -- boot 1 produced no arm end line, so order 8 is not established and this boot would test nothing"
fi

run mxo9ctl \
  "the allocation-order control: the pilot's storage carve byte for byte, in an order-9 region reached by padding the stack alone" \
  "$A/order9-ctl/r1_slots_pools.dom" c685b0d7a95492ef "$L/m1-order9-ctl.txt" \
  "$A/order9-ctl/qemu-drop/boot.log" \
  "PRIMARY -- attribution of boot 1's wedge, and it is the whole point of this boot. This image keeps M1_MAXRET = 4096, so its storage carve is 76,832 bytes, IDENTICAL to the pilot image that ran to target twice; it reaches allocation order 9 -- the order the wedging image had -- by padding R1_STACK to 1 MiB and nothing else. WEDGES means the REGION SIZE is the trigger and the storage carve is innocent, which also means M1_MAXRET is innocent by association and the pressure arm's ceiling is set by the order rather than by the buffer. COMPLETES means the region size is fine at order 9 and the trigger is the storage carve total, so boot 1's wedge is about a megabyte of globals. SECONDARY, and free either way if it completes: drop and ring at the full 10C = 655,320 give a REPLICATION of boot 2's curves at a different region size. It is a replication and NOT the protocol's repetition 2, because the image differs from boot 2's; label it as such. A curve that moves with the region size would itself be the finding, since the earlier capacity work concluded that residency displaces nodes and address range buys nothing. RISK CARRIED KNOWINGLY: 1,310,640 allocations at the measured bound c < 0.0498 is about 65,270 distinct indices against a 65,532 pool -- boot 2 did the same and did not exhaust, but exhaustion would trap at cause 30 and, with mtvec 0, look like a hang. $COMMON"

echo "CHAIN_MAXRET_DONE"
