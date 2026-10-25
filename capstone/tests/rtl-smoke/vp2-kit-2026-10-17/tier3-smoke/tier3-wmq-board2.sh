#!/bin/bash
# wmem vp2-nested-pools with the Q-15 repair on BOARD 2, tree B (2026-10-19 evening; pre-registered in README.md before the
# boot). The compiler lane's invocation (wm1's with only the image builds changed): run-defects.py --arm virtual-nested-pools
# on t3-new9's cases-ON / controls-ON, through tier3-appvm-arm.sh with EXPECT_IMAGES = wmq1.images.SHA256SUMS (dae7f5a871e0e477),
# on kit bkit7-rev2xBD (tree B, board 2, module 39d8492d, the S-23 retry monitor f13dbc5, dev's lease-client driver).
# Scored by compare-pilot.py against run C (t3-wmq-C/out), which C' on the board module's QEMU twin a5dec6a3 read 23/23
# identical. Waits for <after> to read CHAIN_B2 DONE or HOLD. Usage: tier3-wmq-board2.sh <after> <out>. Ends by itself.
# WMQ_ARM / WMQ_BUSY / WMQ_ROOT stub for the dry run.
main() {
  set -u
  local W=${1:?after} O=${2:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone
  local K=$H/t3-smoke/bkit7-rev2xBD R=${WMQ_ROOT:-$H/t3-run-treeB/wmq} I=$H/t3-new9
  local RUN=/tmp/capstone/wt-b2/capstone/bug-corpora/wireshark/wmem-repros/shared/run-defects.py
  until grep -a -q "^CHAIN_B2 DONE\|^CHAIN_B2 HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^CHAIN_B2 DONE\|^CHAIN_B2 HOLD" "$W" || { echo "WMQ HOLD: the board-2 chain did not end" >> "$O"; return 1; }
  [ "$(sha256sum $RUN | cut -c1-16)" = 780a0975142e2221 ] || { echo "WMQ STOP: runner" >> "$O"; return 1; }
  [ "$(sha256sum $I/wmq1.images.SHA256SUMS | cut -c1-16)" = dae7f5a871e0e477 ] || { echo "WMQ STOP: image list" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "FPGA_BOARD=2" $K/board.env && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_8de3c48cf" $K/board.env \
    || { echo "WMQ STOP: kit is not bkit7-rev2xBD as recorded" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "WMQ STOP: $R exists" >> "$O"; return 1; }
  ${WMQ_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2} > /dev/null || { echo "WMQ STOP: board 2 is busy" >> "$O"; return 1; }
  echo "WMQ BEGIN $(date +%T)" >> "$O"
  EXPECT_IMAGES=$I/wmq1.images.SHA256SUMS taskset -c 16-31 timeout 10800 ${WMQ_ARM:-bash $S/tier3-appvm-arm.sh} --board-kit $K $R $RUN \
    --arm virtual-nested-pools --hosted-build $I/wmem/cases-ON --controls-hosted-build $I/wmem/controls-ON --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 >> "$O" 2>&1
  echo "WMQ ARM END rc=$? $(date +%T)" >> "$O"
  python3 $S/compare-pilot.py $H/t3-wmq-C/out $R/out > $R.cmp.txt 2>&1; echo "WMQ COMPARE vs run C rc=$? :: $(tail -1 $R.cmp.txt)" >> "$O"
  local B=$R/session/guest/board.txt
  echo "WMQ readings: cause=29 at capstone_cap_derive $(grep -a -c 'cause=29.*capstone_cap_derive' $B 2>/dev/null); collect skipped $(grep -a -c 'collect skipped' $B 2>/dev/null); DRT1 $(grep -a -c '^DRT1:' $B 2>/dev/null) DRTX $(grep -a -c '^DRTX:' $B 2>/dev/null) Oops $(grep -a -c Oops $B 2>/dev/null)" >> "$O"
  echo "WMQ DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
