#!/bin/bash
# cpyp-DB (2026-10-20; pre-registered in README.md before the boot, with NO predicted value), board 2, tree B: cpyp-D with
# ONE change, the bitstream. Kit bkit7-rev2xnBD is bkit7-rev2xnD with tree B's board fields (make-cpyb-kit.sh); module
# 39d8492d and monitor f13dbc5 as cpyp-D. Tree B differs from tree A by S-29's satp flush. Readings as cpyp-DZ's.
# Usage: tier3-cpy-b.sh <board 2's q15 out> <out>: starts once it reads Q15 DONE, STOP or HOLD. Ends by itself.
# CPP_ARM / CPP_BUSY / CPP_ROOT stub.
main() {
  set -u
  local W=${1:?after} O=${2:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone
  local RUN=/tmp/capstone/wt-cpy-meas/capstone/bug-corpora/cpython/allocator-boundary-repros/runners/virtual/run-virtual.py
  local P=/tmp/capstone/land-board/capstone/bug-corpora/cpython/allocator-boundary-repros/results/2026-10-19-vp2-trimmed-pinned
  local IMGD=$H/t3-new7/cpython-lt0-dl a kit mon R B
  until grep -a -q "^Q15 DONE\|^Q15 STOP\|^Q15 HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^Q15 DONE\|^Q15 STOP\|^Q15 HOLD" "$W" || { echo "CPP HOLD: board 2's q15 did not end" >> "$O"; return 1; }
  [ "$(sha256sum $RUN | cut -c1-16)" = b98e90fc86fafe1a ] && [ "$(sha256sum $P/stage-closure.txt | cut -c1-16)" = b41545623090d0cf ] \
    && [ "$(sha256sum $IMGD/image/cpython.dom | cut -c1-16)" = 20bcfdd1073976e0 ] || { echo "CPP STOP: runner, closure or image" >> "$O"; return 1; }
  for a in "cpyp-DB bkit7-rev2xnBD f13dbc5"; do
    read -r tag kit mon <<<"$a"; kit=$H/t3-smoke/$kit; R=${CPP_ROOT:-$H/t3-run-treeB}/$tag
    ( cd $kit && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_MONITOR_COMMIT=$mon" $kit/board.env && grep -q norandmaps $kit/board.env && [ "$(sha256sum $kit/adapter/module/capstone_vm.ko | cut -c1-16)" = 39d8492d5c18216b ] && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_8de3c48cf" $kit/board.env && grep -qx "FPGA_BOARD=2" $kit/board.env \
      || { echo "CPP STOP: $kit is not as recorded (monitor $mon)" >> "$O"; return 1; }
    [ -e "$R" ] && { echo "CPP STOP: $R exists" >> "$O"; return 1; }
    ${CPP_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2} > /dev/null || { echo "CPP STOP: board 2 is busy" >> "$O"; return 1; }
    echo "CPP $tag BEGIN $(date +%T) kit $(basename $kit) monitor $mon" >> "$O"
    taskset -c 16-31 timeout 10800 ${CPP_ARM:-bash $S/tier3-appvm-arm.sh} --board-kit $kit $R $RUN --arm virtual-malloc --build $IMGD \
      --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 --guest-env PYTHONHASHSEED=0 --stdlib-sourceless --timeout 300 \
      --stdlib-closure $P/stage-closure.txt --only 03,17,23 >> "$O" 2>&1
    echo "CPP $tag ARM END rc=$? $(date +%T)" >> "$O"
    B=$R/session/guest/board.txt
    echo "CPP $tag steps: $(grep -a -c '@@STEP-BEGIN' $B 2>/dev/null) begun, $(grep -a -c '@@STEP-END' $B 2>/dev/null) ended; DRT1 $(grep -a -c '^DRT1:' $B 2>/dev/null) DRTX $(grep -a -c '^DRTX:' $B 2>/dev/null) FRTY $(grep -a -c '^FRTY:' $B 2>/dev/null) Oops $(grep -a -c Oops $B 2>/dev/null)" >> "$O"
    echo "CPP $tag pc=0 by step: $(awk '/@@STEP-BEGIN/{s=$2} /domain fault cause=12 pc=0x0/{print s}' $B 2>/dev/null | sort | uniq -c | tr '\n' ' ')" >> "$O"
  done
  echo "CPP DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
