#!/bin/bash
# mruby's stock smoke x4 in one cold boot with the launcher's service trace (trace-smoke.py; 2026-10-21, pre-registered in
# README.md before the boot). The image is the reference's mruby-8c7cd27614353067.dom and its smoke.rb; the session tool
# is a clean wt-tools2 (dev e6c68a531557: the step bound and the stall capture). Booked on the lab hub; refused on a 409.
# Usage: tier3-trace-smoke.sh <board 1|2> <out log>. TS_RUN / TS_BUSY / TS_BOOK / TS_ROOT stub the dry run.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log}
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone K bs R BUSY rc bid
  local REF=$H/corpus-mp/refs/tB-mimic/mruby-stock/session/collect
  local IMG=$REF/images/mruby-8c7cd27614353067.dom RB=$REF/share/mruby-cases/smoke.rb TOOLS=/tmp/capstone/wt-tools2/capstone/bug-corpora/tools
  case $BRD in
    1) K=$H/t3-smoke/bkit7-rev2xDN2; bs=caplifive_vp2c_8522d72fe; R=${TS_ROOT:-$H/t3-run-treeA/trace-smoke}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit7-rev2xBDN2; bs=caplifive_vp2c_8de3c48cf; R=${TS_ROOT:-$H/t3-run-treeB/trace-smoke}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "TS STOP: board $BRD" >> "$O"; return 2;;
  esac
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 54d8a3af1158c3f4 ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } || { echo "TS STOP: kit $K" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/wt-tools2 rev-parse --short=12 HEAD)" = e6c68a531557 ] && [ -z "$(git -C /tmp/capstone/wt-tools2 status --porcelain)" ] || { echo "TS STOP: wt-tools2 is not a clean e6c68a531557" >> "$O"; return 1; }
  [ "$(sha256sum $IMG | cut -c1-16)" = 8c7cd27614353067 ] && [ -s $RB ] || { echo "TS STOP: image or smoke.rb" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "TS STOP: $R exists" >> "$O"; return 1; }
  ${TS_BUSY:-$BUSY} > /dev/null || { echo "TS STOP: board $BRD is busy" >> "$O"; return 1; }
  bid=$(${TS_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 120 "vp2-tier12 mruby trace-smoke (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "TS STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "TS NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "TS BEGIN $(date +%T) board $BRD image 8c7cd27614353067 smoke.rb $(sha256sum $RB | cut -c1-16) tools e6c68a531557 booking ${bid:-none}" >> "$O"
  taskset -c 16-31 timeout 14400 ${TS_RUN:-python3 $S/trace-smoke.py} --board-kit $K --tools $TOOLS --image $IMG --smoke $RB --out $R > $R.log 2>&1; rc=$?
  [ -n "$bid" ] && ${TS_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "TS END rc=$rc $(date +%T) :: $(tail -1 $R.log | cut -c1-160)" >> "$O"
  grep "^STEP " $R.log >> "$O"
  echo "TS board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
