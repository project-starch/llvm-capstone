#!/bin/bash
# cpython allocator-boundary lt0 on the board (2026-10-19 evening; pre-registered in README.md before the boot): the stage
# (dev 81fb4e04d42e: image 20bcfdd1073976e0, stage-closure.txt b41545623090d0cf, run-virtual.py --stdlib-sourceless
# b98e90fc86fafe1a) through tier3-appvm-arm.sh on kit bkit7-rev2xnD (norandmaps, module 39d8492d, the S-23 retry monitor
# f13dbc5), board 1, tree A. Judged by judge-cpy-set.py against prediction.json lt0 (dev 18a7cf883de3's rule).
# Usage: tier3-cpy-lt0.sh <out>. Ends by itself. CPY_ARM / CPY_BUSY / CPY_ROOT stub for the dry run.
main() {
  set -u
  local O=${1:?out} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone
  local K=$H/t3-smoke/bkit7-rev2xnD R=${CPY_ROOT:-$H/t3-run-treeA/cpy-lt0}
  local RUN=/tmp/capstone/wt-cpy-meas/capstone/bug-corpora/cpython/allocator-boundary-repros/runners/virtual/run-virtual.py
  local P=/tmp/capstone/land-board/capstone/bug-corpora/cpython/allocator-boundary-repros/results/2026-10-19-vp2-trimmed-pinned
  local IMGD=$H/t3-new7/cpython-lt0-dl
  [ "$(sha256sum $RUN | cut -c1-16)" = b98e90fc86fafe1a ] || { echo "CPY STOP: runner" >> "$O"; return 1; }
  [ "$(sha256sum $P/stage-closure.txt | cut -c1-16)" = b41545623090d0cf ] || { echo "CPY STOP: closure" >> "$O"; return 1; }
  [ "$(sha256sum $IMGD/image/cpython.dom | cut -c1-16)" = 20bcfdd1073976e0 ] || { echo "CPY STOP: image" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_MONITOR_COMMIT=f13dbc5" $K/board.env && grep -q "norandmaps" $K/board.env \
    && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_8522d72fe" $K/board.env || { echo "CPY STOP: kit is not bkit7-rev2xnD as recorded" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "CPY STOP: $R exists" >> "$O"; return 1; }
  ${CPY_BUSY:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1} > /dev/null || { echo "CPY STOP: board 1 is busy" >> "$O"; return 1; }
  echo "CPY lt0 BEGIN $(date +%T)" >> "$O"
  taskset -c 16-31 timeout 21600 ${CPY_ARM:-bash $S/tier3-appvm-arm.sh} --board-kit $K $R $RUN \
    --arm virtual-malloc --build $IMGD --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 \
    --guest-env PYTHONHASHSEED=0 --stdlib-sourceless --timeout 300 --stdlib-closure $P/stage-closure.txt >> "$O" 2>&1
  echo "CPY lt0 ARM END rc=$? $(date +%T)" >> "$O"
  python3 $S/judge-cpy-set.py $R/out lt0 > $R.judge.txt 2>&1; echo "CPY lt0 JUDGE rc=$? :: $(tail -1 $R.judge.txt)" >> "$O"
  local B=$R/session/guest/board.txt
  echo "CPY lt0 S-23: DRT1 $(grep -a -c '^DRT1:' $B 2>/dev/null) DRTX $(grep -a -c '^DRTX:' $B 2>/dev/null) Oops $(grep -a -c 'Oops' $B 2>/dev/null) (from $B)" >> "$O"
  echo "CPY lt0 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
