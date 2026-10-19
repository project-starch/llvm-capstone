#!/bin/bash
# diag31 (2026-10-20; pre-registered in README.md before the boot): case 31 (fz10, statDecodePage) with the compiler lane's
# DIAG31 line (lane diag31 5876e58b6a1e), on BOARD 1, tree A, kit bkit7-rev2xD, --only 31 with both controls. The line prints
# aData's true bounds, the out-of-window cursor and the LCC-decoded bounds of that pointer just before the program's own
# getVarint32 load. As q15's path: dev's arm script and run-virtual.py from the dev-pinned /tmp/capstone/wt-q15.
# Usage: tier3-diag31-board1.sh <go file> <out>: starts once <go file> reads D31_GO. Ends by itself. D31_ARM / D31_BUSY /
# D31_ROOT stub for the dry run.
main() {
  set -u
  local W=${1:?go} O=${2:?out} end=$(( $(date +%s) + 43200 ))
  local H=/mnt/hdd/$USER/capstone Q=/tmp/capstone/wt-q15/capstone
  local K=$H/t3-smoke/bkit7-rev2xD R=${D31_ROOT:-$H/t3-run/diag31} I=$H/t3-images-nb/diag31.images.SHA256SUMS B=$H/diag31-build-full
  local RUN=$Q/bug-corpora/sqlite/engine-repros/shared/run-virtual.py ARM=$Q/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh rc
  until grep -a -q "^D31_GO" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  grep -a -q "^D31_GO" "$W" || { echo "D31 HOLD: no D31_GO" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/wt-q15 rev-parse --short=12 HEAD)" = f1d522d7fff1 ] && [ -z "$(git -C /tmp/capstone/wt-q15 status --porcelain)" ] \
    && [ "$(sha256sum $RUN | cut -c1-16)" = b3f1aa0ea60442f8 ] && [ "$(sha256sum $ARM | cut -c1-16)" = 666d920eedef445b ] \
    || { echo "D31 STOP: wt-q15, runner or arm script" >> "$O"; return 1; }
  [ "$(sha256sum $I | cut -c1-16)" = e6d4c35af624ad23 ] && grep -q '"extra_patches": "sublet/diag31-statdecodepage.patch"' $B/build.json \
    || { echo "D31 STOP: image list or build" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_8522d72fe" $K/board.env && ! grep -q "^FPGA_BOARD=2" $K/board.env \
    || { echo "D31 STOP: kit is not bkit7-rev2xD as recorded" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "D31 STOP: $R exists" >> "$O"; return 1; }
  ${D31_BUSY:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1} > /dev/null || { echo "D31 STOP: board 1 is busy" >> "$O"; return 1; }
  echo "D31 BEGIN $(date +%T)" >> "$O"
  EXPECT_IMAGES=$I taskset -c 16-31 timeout 10800 ${D31_ARM:-bash $ARM} --board-kit $K $R $RUN --arm virtual-nested-pools --build $B \
    --only 31 --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 >> "$O" 2>&1; rc=$?
  echo "D31 ARM END rc=$rc $(date +%T)" >> "$O"
  local G=$R/session/guest/board.txt
  [ -s "$G" ] || { echo "D31 NO DATA: no guest board.txt" >> "$O"; echo "D31 DONE $(date +%T)" >> "$O"; return 1; }
  grep -a -o "DIAG31 adata=.*" $G | sed 's/^/D31 LINE /' >> "$O"
  echo "D31 readings: DIAG31 lines $(grep -a -c '^DIAG31 ' $G) (anywhere in a line: $(grep -a -c 'DIAG31 adata=' $G)); steps $(grep -a '@@STEP-END' $G | tr '\n' ';'); faults $(grep -a -o 'domain fault cause=[0-9]*' $G | tr '\n' ' '); FRTY $(grep -a -c '^FRTY:' $G) DRT1 $(grep -a -c '^DRT1:' $G) Oops $(grep -a -c Oops $G)" >> "$O"
  echo "D31 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
