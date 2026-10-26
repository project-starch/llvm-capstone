#!/bin/bash
# BITSTREAM 3's regression controls on the flashed board (2026-10-27, before the flash; pre-registered in tier3-smoke/
# README.md). These are the rtl lane's list, item (c): the core differs from bitstream 2 only in the FPU, so each must
# read what it read on bitstream 2. Run only after run-b3-first.sh's first boots read as predicted. Each step is its own
# boot, on the bitstream-3 kit:
#  1. hold 100000 (tier3-hold-batch.sh): HOLD-PEAK and HOLD-OK 100000, exit 0, high_water 105,569 of 262,144;
#  2. R-61 (tier3-r61-bare.sh, kit 61faf96f5863): the control PASS, r61-stale2 and r61-stale NOT-REPRODUCED (the fixed
#     tree's vectors);
#  3. R-61 arming (tier3-r61a-bare.sh, kit 6910f80d4903): the control and stale0 exact, r61a-stale3 and r61a-stale4 on the
#     fixed tree's vectors;
#  4. the fixes kit (tier3-rfix-bare.sh, kit 30117b74cc8f): every image EXACT, so R-58, R-59 and Q-24 read as on
#     bitstream 2.
# Usage: run-b3-after.sh <board 1|2> <out dir>. B3A_HOLD / B3A_R61 / B3A_R61A / B3A_RFIX stub the dry run; B3_KIT overrides
# the kit.
main() {
  set -u
  local BRD=${1:?board} D=${2:?out dir} H=/mnt/hdd/$USER/capstone S
  S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); local T3; T3=$(cd $S/../../tier3-smoke && pwd)
  local K=${B3_KIT:-$H/t3-smoke/bkit11-b3-b$BRD} L
  [ -e $D ] && { echo "B3A STOP: $D exists"; return 1; }
  mkdir -p $D; L=$D/SEQ
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -q "^BOARD_BITSTREAM=caplifive_.*e7af48457" $K/board.env \
    || { echo "B3A STOP: kit $K does not verify or is not a bitstream-3 kit" >> $L; return 1; }
  echo "B3A START $(date +%T) board $BRD kit ${K##*/}" >> $L
  HB_ROOT=$D/hold ${B3A_HOLD:-bash $T3/tier3-hold-batch.sh} $BRD $K - - $L; echo "B3A hold rc=$? $(date +%T)" >> $L
  R61_KIT=$K R61_ROOT=$D/r61-bare ${B3A_R61:-bash $T3/tier3-r61-bare.sh} $BRD $L 61faf96f5863; echo "B3A r61 rc=$? $(date +%T)" >> $L
  R61_KIT=$K R61_ROOT=$D/r61a-bare ${B3A_R61A:-bash $T3/tier3-r61a-bare.sh} $BRD $L 6910f80d4903; echo "B3A r61a rc=$? $(date +%T)" >> $L
  RF_KIT=$K RF_ROOT=$D/rfix-bare ${B3A_RFIX:-bash $T3/tier3-rfix-bare.sh} $BRD $L 30117b74cc8f; echo "B3A rfix rc=$? $(date +%T)" >> $L
  echo "B3A board$BRD DONE $(date +%T)" >> $L
}
main "$@"; exit $?
