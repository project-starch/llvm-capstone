#!/bin/bash
# cpyp-DFp (2026-10-20; pre-registered in README.md before the boot), board 1, tree A: cpyp-DF with ONE change, the image:
# --build t3-cw/p, cpython.dom 7e5ba3b993b98957 = cpython-lt0's recorded build replayed with only the runtime archive changed
# (5b3e6b79 + 232d13b2's heap.c hunks, which remove backend_pointer's out-of-window intermediate; the compiler lane's replay,
# whose control arm reproduces the recorded image byte for byte). Kit, runner, closure and --only as cpyp-DF. Usage:
# tier3-cpy-fp.sh <cpyp-DF out> <out>: starts once cpyp-DF ends (CPP DONE, STOP or HOLD). Ends by itself. CPP_* stub.
main() {
  set -u
  local W=${1:?after} O=${2:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone
  local RUN=/tmp/capstone/wt-cpy-meas/capstone/bug-corpora/cpython/allocator-boundary-repros/runners/virtual/run-virtual.py
  local P=/tmp/capstone/land-board/capstone/bug-corpora/cpython/allocator-boundary-repros/results/2026-10-19-vp2-trimmed-pinned
  local IMGD=$H/t3-cw/p a kit mon R B
  until grep -a -q "^CPP DONE\|^CPP STOP\|^CPP HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^CPP DONE" "$W" || { echo "CPP HOLD: cpyp-DF did not complete ($(tail -1 "$W"))" >> "$O"; return 1; }
  [ "$(sha256sum $RUN | cut -c1-16)" = b98e90fc86fafe1a ] && [ "$(sha256sum $P/stage-closure.txt | cut -c1-16)" = b41545623090d0cf ] \
    && [ "$(sha256sum $IMGD/image/cpython.dom | cut -c1-16)" = 7e5ba3b993b98957 ] || { echo "CPP STOP: runner, closure or image" >> "$O"; return 1; }
  for a in "cpyp-DFp bkit7-rev2xnDF f13dbc5"; do
    read -r tag kit mon <<<"$a"; kit=$H/t3-smoke/$kit; R=${CPP_ROOT:-$H/t3-run-treeA}/$tag
    ( cd $kit && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_MONITOR_COMMIT=$mon" $kit/board.env && grep -q norandmaps $kit/board.env && [ "$(sha256sum $kit/adapter/module/capstone_vm.ko | cut -c1-16)" = 53507c769b51758a ] \
      || { echo "CPP STOP: $kit is not as recorded (monitor $mon)" >> "$O"; return 1; }
    [ -e "$R" ] && { echo "CPP STOP: $R exists" >> "$O"; return 1; }
    ${CPP_BUSY:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1} > /dev/null || { echo "CPP STOP: board 1 is busy" >> "$O"; return 1; }
    echo "CPP $tag BEGIN $(date +%T) kit $(basename $kit) monitor $mon" >> "$O"
    taskset -c 16-31 timeout 10800 ${CPP_ARM:-bash $S/tier3-appvm-arm.sh} --board-kit $kit $R $RUN --arm virtual-malloc --build $IMGD \
      --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 --guest-env PYTHONHASHSEED=0 --stdlib-sourceless --timeout 300 \
      --stdlib-closure $P/stage-closure.txt --only 03,17,23 >> "$O" 2>&1
    echo "CPP $tag ARM END rc=$? $(date +%T)" >> "$O"
    B=$R/session/guest/board.txt
    echo "CPP $tag steps: $(grep -a -c '@@STEP-BEGIN' $B 2>/dev/null) begun, $(grep -a -c '@@STEP-END' $B 2>/dev/null) ended; DRT1 $(grep -a -c '^DRT1:' $B 2>/dev/null) DRTX $(grep -a -c '^DRTX:' $B 2>/dev/null) FRTY $(grep -a -c '^FRTY:' $B 2>/dev/null) Oops $(grep -a -c Oops $B 2>/dev/null); banner x$(grep -a -c 'fault-regs-srz-1' $B 2>/dev/null); FREG blocks $(grep -a -c 'capstone_vm: FREG thread=' $B 2>/dev/null)" >> "$O"
    echo "CPP $tag pc=0 by step: $(awk '/@@STEP-BEGIN/{s=$2} /domain fault cause=12 pc=0x0/{print s}' $B 2>/dev/null | sort | uniq -c | tr '\n' ' ')" >> "$O"
    awk '/@@STEP-BEGIN/{s=$2} /capstone_vm: FREG/{sub(/.*capstone_vm: /,""); print "CPP FREG " s " :: " $0}' $B 2>/dev/null | cut -c1-260 >> "$O"
  done
  echo "CPP DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
