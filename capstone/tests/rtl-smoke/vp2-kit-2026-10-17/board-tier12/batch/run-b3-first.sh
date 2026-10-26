#!/bin/bash
# BITSTREAM 3's first boots on the flashed board (capstone-ariane e7af48457; written 2026-10-27 before the flash,
# pre-registered in tier3-smoke/README.md). Each step is its own cold boot:
#  1. the rtl lane's fp-all bare image (tier3-fpds-bare.sh, kit commit 84488ed6db17) on the bitstream-3 kit. Predicted
#     reading: fpds-grade.py's fixed-RTL branch, "FPDS NOT ON SILICON: 0 fdiv and 0 fsqrt mismatches" (R-62 fixed on
#     silicon). Bitstream 2 read the defect vector on both boards.
#  2. the rtl lane's fp-rm bare image (tier3-fprm-bare.sh, kit commit 75f129f56497). Predicted reading: fprm-grade.py's
#     fixed-RTL branch, "FPRM NOT ON SILICON: the fixed-RTL vector" (R-63 fixed on silicon).
#     Neither step gates the next. A defect vector, VOID or FINDING is surfaced as a result about the build, and the
#     chain still continues. The graders' labels were written for bitstream 2, where "CONFIRMED ON SILICON" meant the
#     defect was present.
#  3. run-b2-first.sh unchanged on the bitstream-3 kit (BF_KIT): s23 bare --expect batch, va-alloc bare --expect b2, then
#     Tier 1 (firmware 0c24496f). The boot path is bitstream 2's, so its predictions are bitstream 2's own readings.
# Usage: run-b3-first.sh <board 1|2> <bitstream name, no .bit> <out dir>. B3_FPDS / B3_FPRM / B3_FIRST stub the dry run;
# B3_KIT overrides the kit.
main() {
  set -u
  local BRD=${1:?board} BS=${2:?bitstream name} D=${3:?out dir} H=/mnt/hdd/$USER/capstone S
  S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); local T3; T3=$(cd $S/../../tier3-smoke && pwd)
  local K=${B3_KIT:-$H/t3-smoke/bkit11-b3-b$BRD} L g
  [ -e $D ] && { echo "B3 STOP: $D exists"; return 1; }
  mkdir -p $D; L=$D/SEQ
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$BS" $K/board.env \
    || { echo "B3 STOP: kit $K does not verify or does not name $BS" >> $L; return 1; }
  echo "B3 START $(date +%T) board $BRD bitstream $BS kit ${K##*/}" >> $L
  RF_KIT=$K RF_ROOT=$D/fpds-bare ${B3_FPDS:-bash $T3/tier3-fpds-bare.sh} $BRD $L 84488ed6db17
  g=$(grep -a '^FPDS ' $L | grep -v ' read: \| want: ' | tail -1)
  echo "B3 FP-ALL: ${g:-NO GRADE LINE} :: $(case "$g" in "FPDS NOT ON SILICON"*) echo 'AS PREDICTED (R-62 fixed on silicon)';; *) echo 'NOT AS PREDICTED';; esac)" >> $L
  RF_KIT=$K RF_ROOT=$D/fprm-bare ${B3_FPRM:-bash $T3/tier3-fprm-bare.sh} $BRD $L 75f129f56497
  g=$(grep -a '^FPRM ' $L | grep -v ' read: \| want ' | tail -1)
  echo "B3 FP-RM: ${g:-NO GRADE LINE} :: $(case "$g" in "FPRM NOT ON SILICON"*) echo 'AS PREDICTED (R-63 fixed on silicon)';; *) echo 'NOT AS PREDICTED';; esac)" >> $L
  BF_KIT=$K ${B3_FIRST:-bash $S/run-b2-first.sh} $BRD $BS $D/first; echo "B3 FIRST rc=$? $(date +%T)" >> $L
  sed 's/^/  /' $D/first/SEQ >> $L 2>/dev/null
  echo "B3 board$BRD DONE $(date +%T)" >> $L
}
main "$@"; exit $?
