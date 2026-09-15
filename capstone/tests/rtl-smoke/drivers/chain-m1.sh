#!/bin/bash
# chain-m1.sh -- the no-reclamation baseline (NOT M1: the reclaiming configuration does not exist),
# FOUR boots over board-r1e4.sh. PREPARED 2026-09-15, NOT LAUNCHED: the lead's call.
#   boots 1-3  the four retention patterns at the 256-entry DIAGNOSTIC capacity, THREE repetitions of each
#              per boot (12 invocations = the M-9 cap) -> NINE runs per pattern over three boots, which is
#              what METHODS' "five measured runs per arm or point, spanning at least three boots" asks of
#              primary FPGA timing. "Diagnostic" here is the CAPACITY, not the platform.
#   boot 4     the DROP pattern alone to the 80 % budget on the deployed table (52,425 allocations):
#              the bound, and a corroborating long curve. One such run fits a boot (the budget IS 80 % of
#              the table), so it is a SHORTENED POINT under METHODS rule 5 with its own configuration
#              label (stop=budget) and it is NOT primary timing. Five runs over three boots for that point
#              would cost five more boots; that is the lead's call, not a default.
# Nothing here approaches exhaustion, which R-12 makes a wedge.
#
#   M1_IMG=<r1_slots_pools.dom built at DOMAIN_BASE_VA=0x410000> M1_HASH=<sha256/16> R1_HOST=<sqlite_host_rr.user> \
#   M1_QEMU_LOG=<the emulator flow check's boot.log, from qemu-m1-flowcheck.sh> [MEMLOCK=<lock>] bash chain-m1.sh
#
# THE IMAGE MUST BE LINKED AT DOMAIN_BASE_VA=0x410000. The prepared build11 (9a01b12a4db639b6) was built at the
# script's DEFAULT 0x10000 for emulator validation only; staged on the board it collides with the k800 control
# at the same VA and preflight C15 refuses the boot. Rebuild at 0x410000, re-run qemu-m1-flowcheck.sh so the
# relocated image earns its OWN qemu-pass record rather than inheriting build11's, and pass that hash as M1_HASH.
# A different toolchain gives a different hash again; the same rule applies.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
U=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/unify; mkdir -p "$U"
L=$R/capstone/tests/rtl-smoke/drivers/lists
: "${M1_IMG:?}" "${M1_HASH:?}" "${M1_QEMU_LOG:?}" "${R1_HOST:?set R1_HOST=<path to sqlite_host_rr.user, the readback host>}"
PREREG="SHAPE: each object is a carved LINEAR leaf handed out as a delin-d NONLIN alias under a REV handle kept in its slot (sublet_carve + sublet_take), released by sublet_give (revoke through the slot handle) -- R1 individual pattern, same primitives. BRACKETS: take_cyc = LDC + MREV + STC + DELIN (sublet_take); give_cyc = LDC handle + REVOKE + LCC type + STC x2 (sublet_give_to), NO fill: the alias-s revoke returns the region LIN (R1 measured fb=0 at 64 B) and init_n stays 0. PRIMARY (what the boots test): take_cyc/n + give_cyc/n is FLAT in cumulative allocations -- a least-squares slope over the snapshots whose span times the run-s allocation range is below 1 % of the mean, and the last quarter-s mean within 1 % of the first quarter-s (minting = a head bump plus one parent link, no walk; release = a childless revoke); a RISE beyond that is the alternative that would matter. SECONDARY (a consistency check assembled from R1-s brackets, scored apart): the sum lands in the band 384-391 raw cycles per allocation -- 384 = rv 51 + fl 12 + in 2 + (reissue_S 355 - reissue_P 36), 391 = R1-s total 593 minus bookkeeping 166 minus the checked use 36, the seven between them being two extra timer reads in R1-s bracket (about 4 cycles at E4-s 2-cycle floor) plus a residual of about 3 between two independently assembled estimates; outside the band = the magnitude model refuted, the primary result unaffected. The split between the two brackets is REPORTED, not predicted (the two STCs sit in give here and in reissue there). The deployed table bounds a run at the same allocation count under every retention pattern (no reclamation: minted = fixture + one per allocation), reached here at the 80 % budget stop; revoked == alloc, retained per arm (drop 0, ring 16, pressure = alloc until the buffer, release 0 after the release), stale_alias_type 2 on silicon (Q-11; 7 on the emulator), live_slot_type 2; every arm stops at stop=target (boot 1) or stop=budget (boot 2)"
cd "$R"
for k in 1 2 3 4; do
  if [ "$k" -le 3 ]; then
    # boots 1-3 index into the 36-line list: the driver takes lines 12*(k-1)+1 .. 12*k
    LIST=$L/m1-diag.txt; SLICE=$k; TAG=r1f6
    DESC="no-reclamation baseline (NOT M1: the reclaiming configuration does not exist), boot $k of 3 at the 256-entry capacity -- DIAGNOSTIC by capacity, not by platform; three repetitions of each of the four retention patterns, nine per pattern over the three boots; the minting and release cost curves as the table fills"
  else
    # boot 4 is ONE long run; its list holds a single line, so it indexes at slice 1
    # a DIFFERENT output tag, or this boot would write into boot 1's directory: never overwrite a run
    LIST=$L/m1-capacity.txt; SLICE=1; TAG=r1f6cap
    DESC="no-reclamation baseline (NOT M1), the deployed table to the 80 % budget with the DROP pattern (a retaining pattern cannot reach it: its own buffer stops it first): the bound, and the high-occupancy end of the release-cost curve -- a SHORTENED POINT under METHODS rule 5 (stop=budget), not primary timing"
  fi
  R1_BOOT=$SLICE R1_OUT_TAG=$TAG R1_BOOT_TAG=sw8x-f6 R1_BOOT_DESC="$DESC" R1_PREREG="$PREREG" \
  R1_IMG=$M1_IMG R1_HASH=$M1_HASH R1_HOST=$R1_HOST R1_LIST=$LIST R1_QEMU_LOG=$M1_QEMU_LOG R1_QEMU_GATE='R1 m1 end' R1_QEMU_GATE_MIN=1 \
    bash "$R/capstone/tests/rtl-smoke/drivers/board-r1e4.sh" > "$U/board-r1f6-b$k.console" 2>&1
  echo "R1F6_B${k}_EXIT=$?" >> "$U/board-r1f6-b$k.console"
  echo "$(date +%T) baseline boot $k: $(tail -1 "$U/board-r1f6-b$k.console")"
done
echo CHAIN_F6_DONE
