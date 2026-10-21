#!/bin/bash
# The SIGALRM-storm matched pair (trace-pair.py; 2026-10-21, pre-registered in README.md before the boot): mruby's stock
# smoke with the service trace, on the r60 image 1ab24a92 (MRB_TICK_UNIT 100) x3 and then f8594e11 (MRB_TICK_UNIT 4) x2,
# in one cold boot. The two builds differ in task_hal.o and task.o only (124 objects, identical sources). Session tool:
# a clean wt-tools2 (dev e6c68a531557). Booked on the lab hub; refused on a 409.
# Usage: tier3-trace-pair.sh <board 1|2> <out log>. TP_RUN / TP_BUSY / TP_BOOK / TP_ROOT stub the dry run.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log}
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone K bs R BUSY rc bid
  local RB=$H/corpus-mp/refs/tB-mimic/mruby-stock/session/collect/share/mruby-cases/smoke.rb TOOLS=/tmp/capstone/wt-tools2/capstone/bug-corpora/tools
  local T100=$H/corpus-mp-r60t/cw/mruby-stock/src/mruby/build/capstone/bin/mruby T4=$H/corpus-mp-r60/cw/mruby-stock/src/mruby/build/capstone/bin/mruby
  case $BRD in
    1) K=$H/t3-smoke/bkit7-rev2xDN2; bs=caplifive_vp2c_8522d72fe; R=${TP_ROOT:-$H/t3-run-treeA/trace-pair}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit7-rev2xBDN2; bs=caplifive_vp2c_8de3c48cf; R=${TP_ROOT:-$H/t3-run-treeB/trace-pair}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "TP STOP: board $BRD" >> "$O"; return 2;;
  esac
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 54d8a3af1158c3f4 ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } || { echo "TP STOP: kit $K" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/wt-tools2 rev-parse --short=12 HEAD)" = e6c68a531557 ] && [ -z "$(git -C /tmp/capstone/wt-tools2 status --porcelain)" ] || { echo "TP STOP: wt-tools2 is not a clean e6c68a531557" >> "$O"; return 1; }
  [ "$(sha256sum $T100 | cut -c1-16)" = 1ab24a9215078a53 ] && [ "$(sha256sum $T4 | cut -c1-16)" = f8594e1167b60307 ] && [ -s $RB ] || { echo "TP STOP: an image or smoke.rb" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "TP STOP: $R exists" >> "$O"; return 1; }
  ${TP_BUSY:-$BUSY} > /dev/null || { echo "TP STOP: board $BRD is busy" >> "$O"; return 1; }
  bid=$(${TP_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 120 "vp2-tier12 mruby tick pair (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "TP STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "TP NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "TP BEGIN $(date +%T) board $BRD t100 1ab24a9215078a53 t4 f8594e1167b60307 smoke.rb $(sha256sum $RB | cut -c1-16) tools e6c68a531557 booking ${bid:-none}" >> "$O"
  taskset -c 16-31 timeout 14400 ${TP_RUN:-python3 $S/trace-pair.py} --board-kit $K --tools $TOOLS --smoke $RB \
    --arm t100=$T100 --arm t100=$T100 --arm t100=$T100 --arm t4=$T4 --arm t4=$T4 --out $R > $R.log 2>&1; rc=$?
  [ -n "$bid" ] && ${TP_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "TP END rc=$rc $(date +%T) :: $(tail -1 $R.log | cut -c1-160)" >> "$O"
  grep "^STEP " $R.log >> "$O"
  echo "TP board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
