#!/bin/bash
# cpython allocator-boundary lt0 on the REBUILT image (2026-10-20, the corpus pass; pre-registered in README.md before its
# boots): image 4641b94cae16a757 (the CAPI build without +movc-keeps-integer-source, -dl), through dev d0f0f4b41ae6's
# tier3-appvm-arm.sh (666d920eedef445b) and run-virtual.py (dc606bd178c10780) from the pinned worktree
# /tmp/capstone/wt-cpy-pass, closure b41545623090d0cf, --stdlib-sourceless, PYTHONHASHSEED=0, norandmaps; kit
# bkit7-rev2xnDN2 (board 1, tree A) or bkit7-rev2xnBDN2 (board 2, tree B), module 54d8a3af. Judged by judge-cpy-set.py
# against the compiler lane's lt0-prediction.json (34aec9edf4471196: r1, mn and the silicon-mimic reference, the old
# record's varying cells kept varying). tier3-cpy-lt0.sh's shape plus the board argument, the hub booking (refused on a
# 409), and the pc=0 and S-23 readings. Usage: tier3-cpy-nm.sh <board 1|2> <after> <tag> <out>: starts when <after> reads
# "PASS <tag> DONE", holds on a PASS STOP or HOLD. CPY_ARM / CPY_BUSY / CPY_ROOT / CPY_BOOK stub the dry run.
main() {
  set -u
  local BRD=${1:?board} W=${2:?after} TAG=${3:?after tag} O=${4:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone P=/mnt/hdd/$USER/capstone/t3-images-cw/lt0-prediction.json
  local WT=/tmp/capstone/wt-cpy-pass; local CR=$WT/capstone/bug-corpora/cpython/allocator-boundary-repros
  local RUN=$CR/runners/virtual/run-virtual.py ARM=$WT/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh
  local CL=$CR/results/2026-10-19-vp2-trimmed-pinned/stage-closure.txt IMGD=$H/t3-nomovc/cpython-lt0-dl K bs R BUSY rc bid
  case $BRD in
    1) K=$H/t3-smoke/bkit7-rev2xnDN2; bs=caplifive_vp2c_8522d72fe; R=${CPY_ROOT:-$H/t3-run-treeA/cpy-nm}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit7-rev2xnBDN2; bs=caplifive_vp2c_8de3c48cf; R=${CPY_ROOT:-$H/t3-run-treeB/cpy-nm}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "CPY STOP: board $BRD" >> "$O"; return 2;;
  esac
  until grep -a -q "^PASS $TAG DONE\|^PASS STOP\|^PASS HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  grep -a -q "^PASS $TAG DONE" "$W" || { echo "CPY HOLD: $TAG did not end DONE ($(tail -1 "$W" 2>/dev/null))" >> "$O"; return 1; }
  [ "$(git -C $WT rev-parse --short=12 HEAD)" = d0f0f4b41ae6 ] && [ -z "$(git -C $WT status --porcelain)" ] || { echo "CPY STOP: $WT is not a clean d0f0f4b41ae6" >> "$O"; return 1; }
  [ "$(sha256sum $RUN | cut -c1-16)" = dc606bd178c10780 ] && [ "$(sha256sum $ARM | cut -c1-16)" = 666d920eedef445b ] || { echo "CPY STOP: runner or arm script" >> "$O"; return 1; }
  [ "$(sha256sum $CL | cut -c1-16)" = b41545623090d0cf ] || { echo "CPY STOP: closure" >> "$O"; return 1; }
  [ "$(sha256sum $IMGD/image/cpython.dom | cut -c1-16)" = 4641b94cae16a757 ] || { echo "CPY STOP: image" >> "$O"; return 1; }
  [ "$(sha256sum $P | cut -c1-16)" = 34aec9edf4471196 ] || { echo "CPY STOP: prediction" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env && grep -q norandmaps $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 54d8a3af1158c3f4 ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } \
    || { echo "CPY STOP: kit $K is not the board-$BRD norandmaps 54d8a3af kit" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "CPY STOP: $R exists" >> "$O"; return 1; }
  ${CPY_BUSY:-$BUSY} > /dev/null || { echo "CPY STOP: board $BRD is busy" >> "$O"; return 1; }
  bid=$(${CPY_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 300 "vp2-tier12 pass cpython lt0 (rebuilt image 4641b94c, board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "CPY STOP: the hub refused the booking (409, someone holds board $BRD): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "CPY NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "CPY nm BEGIN $(date +%T) board $BRD booking ${bid:-none}" >> "$O"
  taskset -c 16-31 timeout 21600 ${CPY_ARM:-bash $ARM} --board-kit $K $R $RUN \
    --arm virtual-malloc --build $IMGD --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 \
    --guest-env PYTHONHASHSEED=0 --stdlib-sourceless --timeout 300 --stdlib-closure $CL >> "$O" 2>&1; rc=$?
  [ -n "$bid" ] && ${CPY_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "CPY nm ARM END rc=$rc $(date +%T)" >> "$O"
  python3 $S/judge-cpy-set.py $R/out lt0 $P > $R.judge.txt 2>&1; echo "CPY nm JUDGE rc=$? :: $(tail -1 $R.judge.txt)" >> "$O"
  local B=$R/session/guest/board.txt
  echo "CPY nm readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B 2>/dev/null) DRT1 $(grep -a -c '^DRT1:' $B 2>/dev/null) DRTX $(grep -a -c '^DRTX:' $B 2>/dev/null) FRTY $(grep -a -c 'FRTY:' $B 2>/dev/null) Oops $(grep -a -c Oops $B 2>/dev/null) banner node-floor-2 x$(grep -a -c node-floor-2 $B 2>/dev/null)" >> "$O"
  echo "CPY nm DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
