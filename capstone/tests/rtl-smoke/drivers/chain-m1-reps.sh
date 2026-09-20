#!/bin/bash
# chain-m1-reps.sh -- M1's repetitions 2 and 3, four boots, and the images are held FIXED on purpose.
#
# WHY THE SAME IMAGES. give_cyc/n was shown on 2026-09-19 to be a property of GLOBAL PLACEMENT and not
# of the arm: three images carrying the IDENTICAL pressure arm read 71.052, 101.049 and 130.041, while
# within one image the cost is the same whatever the arm and reproduces across boots. So a repetition on
# a different image measures layout, not repeatability, and would not be a repetition at all. That is
# also why c685b0d7a95492ef's drop/ring stay labelled a REPLICATION rather than repetition 2.
#
# WHY FOUR BOOTS AND NOT EIGHT. Each list already carries more than one arm:
#     m1-series-nopressure.txt on 1b7a04fe237e1580  ->  drop, ring        (rep 1 = 2026-09-18 boot 2)
#     m1-maxret43296.txt       on 249cfda958f22f16  ->  pressure, release (rep 1 = 2026-09-19 boot 1)
# Two boots of each therefore complete 3 of 3 for all four arms.
#
# WHY NOT chain-r1.sh. That driver slices TWELVE list lines per boot (FROM=$(( 12*(R1_BOOT-1)+1 ))), so
# a three-line list gives boot 2 nothing at all and the boot would run the controls and no arm.
#
#   REPS_HOST=<sqlite_host_rr.user> bash chain-m1-reps.sh
set -u
R=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
U=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/unify; mkdir -p "$U"
L=$R/capstone/tests/rtl-smoke/drivers/lists
A=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/m1-repro
: "${REPS_HOST:?set REPS_HOST=<path to sqlite_host_rr.user, the readback host>}"

COMMON="M1 repetitions on the protocol's terms: three per arm, one per boot, SAME IMAGE as repetition 1, because give_cyc/n is a property of global placement and a repetition on another image would measure layout instead. PRE-REGISTERED READING: take_cyc/n reproduces within the image -- 72.03 drop, 72.01 ring, 72.01 pressure -- and give_cyc/n likewise now that the layout is held; the spread ACROSS boots of one image is the repeatability number the protocol wants. A wide spread is the finding, and the bar is set by the 2026-09-18 against 2026-09-19 comparison, where 145 to 158 of 159 snapshots were identical and the rest differed by under a tenth of a cycle per allocation. CONTROLS: k800 retval=4 first and last; the start line's maxret must equal the built image; minted minus revoked equals 31 on every snapshot. EXPECT RETIREMENT AND DO NOT LOG IT AS A LEAK DEFECT. Monitor 2dcd3a5."

run() { # tag img hash list desc
  R1_BOOT=1 R1_OUT_TAG=$1 R1_BOOT_TAG=sw8x-rp R1_BOOT_DESC="$5" R1_PREREG="$COMMON" \
  R1_IMG=$2 R1_HASH=$3 R1_HOST=$REPS_HOST R1_LIST=$4 \
  R1_QEMU_LOG=$6 R1_QEMU_GATE='R1 m1 end' R1_QEMU_GATE_MIN=1 \
    bash "$R/capstone/tests/rtl-smoke/drivers/board-r1e4.sh" > "$U/board-$1.console" 2>&1
  echo "$(date +%T) $1 EXIT=$?"
}

cd "$R"
for rep in 2 3; do
  run "reps-dr$rep" "$A/../m1-pilot/build/r1_slots_pools.dom" 1b7a04fe237e1580 \
      "$L/m1-series-nopressure.txt" "drop and ring, repetition $rep of 3, on the image repetition 1 used" \
      "$A/../m1-pilot/qemu/qemu-drop/boot.log"
  run "reps-pr$rep" "$A/maxret43296/r1_slots_pools.dom" 249cfda958f22f16 \
      "$L/m1-maxret43296.txt" "pressure and release, repetition $rep of 3, on the image repetition 1 used" \
      "$A/maxret43296/qemu-pressure/boot.log"
done
echo CHAIN_M1_REPS_DONE
