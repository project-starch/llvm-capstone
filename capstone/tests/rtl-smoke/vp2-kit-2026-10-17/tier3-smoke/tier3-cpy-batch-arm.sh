#!/bin/bash
# tier3-cpy-batch.sh generalised to an arm tag (lt1 | lt0), its only change: lt0 is image t3-nomovc-r60/cpython-lt0-dl
# (23a2f0cbd80ddfbe), --arm virtual-malloc, reference batch-refs-r60/ref-batch/lt0 (2026-10-22; each run pre-registered in
# README.md). The original header follows.
# cpython allocator-boundary lt1 (nested pools) on the BATCH bitstream c721c1a2a (2026-10-22; pre-registered in README.md
# before its boot). tier3-cpy-nm.sh's shape with what the batch changes:
#   - the kit is the flashed board's NORANDMAPS batch kit bkit8-batch-nb<board> (make-batch-kit.sh MK_N=1: c721c1a2a,
#     batch module 21fbfc39, monitor d9da397, BOARD_DTB_BOOTARGS norandmaps);
#   - the image is the compiler lane's lt1 build t3-nomovc-r60/cpython-lt1-dl (916e14f9b3f622c9), --arm virtual-nested-pools;
#   - the reference is that lane's batch run batch-refs-r60/ref-batch/lt1 on qkit7-batch2-nr (stamp: runner dc606bd1, tree
#     d0f0f4b41ae6, appvm-arm 666d920e, SWITCHES norandmaps=ON movc_null_scalar=OFF ... generations=ON);
#   - graded by verdicts-compare.py on (verdict, attribution, cause, symbol, offset), not judge-cpy-set.py (lt0 only).
# The runner and arm script come from /tmp/capstone/wt-cpy-pass (clean d0f0f4b41ae6), pinned by those hashes, with the
# lt0 board path's closure (b4154562), --stdlib-sourceless and PYTHONHASHSEED=0, as the reference ran them.
# Usage: tier3-cpy-batch-arm.sh <board 1|2> <out log> <lt1|lt0>. CB_ARM / CB_BUSY / CB_BOOK / CB_ROOT stub the dry run.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log} TAG=${3:?lt1 or lt0} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone REF=/mnt/hdd/$USER/capstone/batch-refs-r60/ref-batch/$TAG PB=/mnt/hdd/$USER/capstone/batch-refs-r60/PASS-B
  local WT=/tmp/capstone/wt-cpy-pass; local CR=$WT/capstone/bug-corpora/cpython/allocator-boundary-repros
  local RUN=$CR/runners/virtual/run-virtual.py ARM=$WT/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh
  local CL=$CR/results/2026-10-19-vp2-trimmed-pinned/stage-closure.txt IMGD=$H/t3-nomovc-r60/cpython-$TAG-dl K R BUSY rc bid ARMN ISHA
  case $TAG in lt1) ARMN=virtual-nested-pools; ISHA=916e14f9b3f622c9;; lt0) ARMN=virtual-malloc; ISHA=23a2f0cbd80ddfbe;; *) echo "CB STOP: tag $TAG" >> "$O"; return 2;; esac
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "CB STOP: board $BRD" >> "$O"; return 2;;
  esac
  K=$H/t3-smoke/bkit8-batch-nb$BRD; R=${CB_ROOT:-$H/t3-run-batch-c721c1a2a-b$BRD/cpy-$TAG}
  [ "$(git -C $WT rev-parse --short=12 HEAD)" = d0f0f4b41ae6 ] && [ -z "$(git -C $WT status --porcelain)" ] || { echo "CB STOP: $WT is not a clean d0f0f4b41ae6" >> "$O"; return 1; }
  [ "$(sha256sum $RUN | cut -c1-16)" = dc606bd178c10780 ] && [ "$(sha256sum $ARM | cut -c1-16)" = 666d920eedef445b ] || { echo "CB STOP: runner or arm script" >> "$O"; return 1; }
  [ "$(sha256sum $CL | cut -c1-16)" = b41545623090d0cf ] || { echo "CB STOP: closure" >> "$O"; return 1; }
  [ "$(sha256sum $IMGD/image/cpython.dom | cut -c1-16)" = $ISHA ] || { echo "CB STOP: image" >> "$O"; return 1; }
  grep -q "^STAMP $TAG runner dc606bd178c10780 tree d0f0f4b41ae6 dirty 0 .* image $ISHA .*appvm-arm 666d920eedef445b" $PB \
    && grep -q "^SWITCHES $TAG .* norandmaps=ON movc_null_scalar=OFF .*generations=ON" $PB \
    && grep -q "^$TAG batch rc=0 .*APPVM read rc=0 " $PB && [ -s $REF/out/verdicts.jsonl ] || { echo "CB STOP: the $TAG reference's stamp, switches or read" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_c721c1a2a" $K/board.env && grep -q norandmaps $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 21fbfc397e7c7a42 ] || { echo "CB STOP: $K is not the norandmaps batch kit" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "CB STOP: $R exists" >> "$O"; return 1; }
  mkdir -p "$(dirname $R)"
  ${CB_BUSY:-$BUSY} > /dev/null || { echo "CB STOP: board $BRD is busy" >> "$O"; return 1; }
  bid=$(${CB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 300 "vp2 batch cpython $TAG (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "CB STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "CB NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "CB $TAG BEGIN $(date +%T) board $BRD kit ${K##*/} booking ${bid:-none}" >> "$O"
  taskset -c 16-31 timeout 21600 ${CB_ARM:-bash $ARM} --board-kit $K $R $RUN \
    --arm $ARMN --build $IMGD --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 \
    --guest-env PYTHONHASHSEED=0 --stdlib-sourceless --timeout 300 --stdlib-closure $CL > $R.run.log 2>&1; rc=$?
  [ -n "$bid" ] && ${CB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "CB $TAG ARM END rc=$rc $(date +%T) :: $(grep '^APPVM read' $R.run.log | tail -1 | cut -c1-200)" >> "$O"
  python3 $S/verdicts-compare.py $REF/out $R/out > $R.cmp.txt 2>&1; echo "CB $TAG COMPARE vs reference rc=$? :: $(tail -1 $R.cmp.txt)" >> "$O"
  local B=$R/session/guest/board.txt
  echo "CB $TAG readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B 2>/dev/null) DRT1 $(grep -a -c '^DRT1:' $B 2>/dev/null) FRTY $(grep -a -c 'FRTY:' $B 2>/dev/null) FRTC $(grep -a -c 'FRTC' $B 2>/dev/null) Oops $(grep -a -c Oops $B 2>/dev/null) INJS $(grep -a -c INJS $B 2>/dev/null) INJL $(grep -a -c INJL $B 2>/dev/null) collect-skipped $(grep -a -c 'collect skipped' $B 2>/dev/null) norandmaps-in-cmdline $(grep -a -c 'norandmaps' $B 2>/dev/null)" >> "$O"
  echo "CB $TAG DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
