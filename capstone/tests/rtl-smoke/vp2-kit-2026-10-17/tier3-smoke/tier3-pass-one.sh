#!/bin/bash
# The corpus pass on the rebuilt images, a WHOLE appvm arm in ONE boot (2026-10-20; pre-registered in README.md before its
# boot): tier3-pass-appvm.sh's gates, booking and readings with the runner given as an argument and no --only split.
# Used for arms whose payload fits one boot (wm0: 23 cases ran in one boot on tree A, 2026-10-19). The runner (path relative
# to wt-gate) and tier3-appvm-arm.sh are pinned to the reference's stamp by sha256/16, the collected images to the
# reference's session/collect/images, and the boot is scored by compare-pilot.py against the reference's whole bundle.
# Usage: tier3-pass-one.sh <board 1|2> <tag> <ref dir> <runner rel path> <runner sha16> <arm sha16> <out> -- <runner args...>
# PASS_ARM / PASS_BUSY / PASS_ROOT / PASS_BOOK stub the dry run. Ends by itself.
main() {
  set -u
  local BRD=${1:?board} TAG=${2:?tag} REF=${3:?ref dir} RREL=${4:?runner path} RSHA=${5:?runner sha} ASHA=${6:?arm sha} O=${7:?out}
  shift 7; [ "${1:-}" = -- ] && shift
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone G=/tmp/capstone/wt-gate K L R BUSY bs
  local ARM=$G/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh
  local RUN=$G/$RREL
  case $BRD in
    1) K=$H/t3-smoke/bkit7-rev2xDN2; bs=caplifive_vp2c_8522d72fe; R=${PASS_ROOT:-$H/t3-run-treeA/pass-$TAG}; BUSY="python3 $G/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit7-rev2xBDN2; bs=caplifive_vp2c_8de3c48cf; R=${PASS_ROOT:-$H/t3-run-treeB/pass-$TAG}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "PASS STOP: board $BRD" >> "$O"; return 2;;
  esac
  L=$R-lists
  [ "$(sha256sum $RUN | cut -c1-16)" = "$RSHA" ] && [ "$(sha256sum $ARM | cut -c1-16)" = "$ASHA" ] \
    || { echo "PASS STOP: runner or arm script is not the reference's ($RSHA / $ASHA)" >> "$O"; return 1; }
  grep -q "runner $RREL $RSHA .*appvm-arm $ASHA" $REF.stamp 2>/dev/null || { echo "PASS STOP: $REF.stamp does not name $RSHA and $ASHA" >> "$O"; return 1; }
  grep -q "SWITCHES movc_null_scalar=1 compact_rounding=ON" $REF.stamp || { echo "PASS STOP: $REF is not a silicon-mimic reference" >> "$O"; return 1; }
  [ -s $REF/out/verdicts.jsonl ] && [ -d $REF/session/collect/images ] || { echo "PASS STOP: no reference bundle or collected images" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 54d8a3af1158c3f4 ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } \
    || { echo "PASS STOP: kit $K is not the board-$BRD 54d8a3af kit" >> "$O"; return 1; }
  mkdir -p $L
  ( cd $REF/session/collect/images && sha256sum $(ls | sort) ) > $L/$TAG.images.SHA256SUMS
  [ -s $L/$TAG.images.SHA256SUMS ] || { echo "PASS STOP: the reference collected no images" >> "$O"; return 1; }
  local t=$TAG rc bid g=""
  [ -e "$R" ] && { echo "PASS STOP: $R exists" >> "$O"; return 1; }
  ${PASS_BUSY:-$BUSY} > /dev/null || { echo "PASS STOP: board $BRD is busy" >> "$O"; return 1; }
  bid=$(${PASS_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 60 "vp2-tier12 pass $t (rebuilt images, board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "PASS STOP: the hub refused the booking (409, someone holds board $BRD): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "PASS NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "PASS $t BEGIN $(date +%T) board $BRD whole arm, $(wc -l < $L/$t.images.SHA256SUMS) images, booking ${bid:-none}" >> "$O"
  EXPECT_IMAGES=$L/$t.images.SHA256SUMS taskset -c 16-31 timeout 10800 ${PASS_ARM:-bash $ARM} --board-kit $K $R $RUN \
    "$@" --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 >> "$O" 2>&1; rc=$?
  [ -n "$bid" ] && ${PASS_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "PASS $t ARM END rc=$rc $(date +%T)" >> "$O"
  case $rc in 78|2) echo "PASS STOP: the arm script refused $t (rc=$rc)" >> "$O"; return 1;; esac
  python3 $S/compare-pilot.py $REF/out $R/out > $R.cmp.txt 2>&1; echo "PASS $t COMPARE vs reference rc=$? :: $(tail -1 $R.cmp.txt)" >> "$O"
  local GB=$R/session/guest/board.txt
  if [ -s "$GB" ]; then
    echo "PASS $t readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $GB) DRT1 $(grep -a -c '^DRT1:' $GB) DRTX $(grep -a -c '^DRTX:' $GB) FRTY $(grep -a -c 'FRTY:' $GB) Oops $(grep -a -c Oops $GB) banner node-floor-2 x$(grep -a -c node-floor-2 $GB)" >> "$O"
  else echo "PASS $t readings: NO DATA, no guest board.txt" >> "$O"; fi
  echo "PASS $TAG DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
