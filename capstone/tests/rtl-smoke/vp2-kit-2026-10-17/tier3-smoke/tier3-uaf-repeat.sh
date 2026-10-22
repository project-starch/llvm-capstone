#!/bin/bash
# The Juliet uaf control repeated in one cold boot per board (uaf-repeat.py; pre-registered in README.md before its
# first boot, 2026-10-22). Tree A on board 1, tree B on board 2, the kits, controls.dom (b1decb06) and juliet-timeout
# (bb65b77a) of the Juliet spatial run. Usage: tier3-uaf-repeat.sh <board 1|2> <out log> [N, default 120].
# UR_RUN / UR_BUSY / UR_BOOK / UR_ROOT stub the dry run.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log} N=${3:-120}
  local H=/mnt/hdd/$USER/capstone K bs R BUSY C bid rc
  case $BRD in
    1) K=$H/t3-smoke/bkit7-rev2xDN2; bs=caplifive_vp2c_8522d72fe; R=${UR_ROOT:-$H/t3-run-treeA/uaf-repeat}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit7-rev2xBDN2; bs=caplifive_vp2c_8de3c48cf; R=${UR_ROOT:-$H/t3-run-treeB/uaf-repeat}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "UR STOP: board $BRD" >> "$O"; return 2;;
  esac
  C=$H/t3-run-treeB/juliet-spatial/run/batch000/stage
  [ "$(sha256sum $C/images/controls.dom | cut -c1-16)" = b1decb0612856121 ] && [ "$(sha256sum $C/juliet-timeout | cut -c1-16)" = bb65b77ac802a9f3 ] \
    || { echo "UR STOP: controls.dom or juliet-timeout is not the Juliet run's" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/wt-tools2 rev-parse --short=12 HEAD)" = e6c68a531557 ] && [ -z "$(git -C /tmp/capstone/wt-tools2 status --porcelain)" ] \
    || { echo "UR STOP: wt-tools2 is not a clean e6c68a531557" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 54d8a3af1158c3f4 ] \
    && [ "$(sha256sum $K/adapter/capstone-vexec | cut -c1-16)" = b27afefc09c6cae6 ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } || { echo "UR STOP: kit $K" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "UR STOP: $R exists" >> "$O"; return 1; }
  ${UR_BUSY:-$BUSY} > /dev/null || { echo "UR STOP: board $BRD is busy" >> "$O"; return 1; }
  bid=$(${UR_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 75 "vp2-tier12 uaf-control repeat (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "UR STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "UR NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "UR BEGIN $(date +%T) board $BRD booking ${bid:-none} N=$N" >> "$O"
  taskset -c 16-31 timeout 7200 ${UR_RUN:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/uaf-repeat.py} \
    --board-kit $K --tools /tmp/capstone/wt-tools2/capstone/bug-corpora/tools --controls $C/images/controls.dom --helper $C/juliet-timeout \
    --n $N --out $R > $R.log 2>&1; rc=$?
  [ -n "$bid" ] && ${UR_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "UR END rc=$rc $(date +%T)" >> "$O"
  grep -a "^BOUNDS-0\|^RUN \|^TALLY\|^completed=\|^NO DATA" $R.log | cut -c1-240 >> "$O"
  local tb=$R/work/board.txt c
  c="pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $tb 2>/dev/null) DRT1 $(grep -a -c '^DRT1:' $tb 2>/dev/null) FRTY $(grep -a -c 'FRTY:' $tb 2>/dev/null) Oops $(grep -a -c Oops $tb 2>/dev/null) INJS $(grep -a -c INJS $tb 2>/dev/null) INJL $(grep -a -c INJL $tb 2>/dev/null)"
  echo "UR readings: $c" >> "$O"
  echo "UR board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
