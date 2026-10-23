#!/bin/bash
# BATCH version (2026-10-23; pre-registered in README.md before its boot): tier3-uaf-repeat.sh on the board's batch kit
# bkit8-batch-b<board> (c721c1a2a, module 21fbfc39, launcher b27afefc), output to
# <mnt-hdd>/capstone/t3-run-batch-c721c1a2a-b<board>/uaf-repeat<tag>. The original header follows.
# The Juliet uaf control repeated in one cold boot per board (uaf-repeat.py; pre-registered in README.md before its
# first boot, 2026-10-22). Tree A on board 1, tree B on board 2, the kits, controls.dom (b1decb06) and juliet-timeout
# (bb65b77a) of the Juliet spatial run. Usage: tier3-uaf-repeat.sh <board 1|2> <out log> [N, default 120].
# UR_RUN / UR_BUSY / UR_BOOK / UR_ROOT stub the dry run. UR_TAG suffixes the run root (uaf-repeat<tag>) and UR_ARGS adds
# uaf-repeat.py options (--no-trace, --dmesg); both empty by default, which is the first run's configuration.
# UR_VEXSHA is the kit launcher's sha256 prefix (default b27afefc09c6cae6). UR_BOOKMIN sets the booking's minutes (default 75). UR_KITSFX picks a kit copy (bkit7-...<sfx>) and UR_MODSHA its module's sha256 prefix (default 54d8a3af1158c3f4).
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log} N=${3:-120}
  local H=/mnt/hdd/$USER/capstone K bs R BUSY C bid rc
  case $BRD in
    1) K=$H/t3-smoke/bkit8-batch-b1; bs=caplifive_vp2c_c721c1a2a; R=${UR_ROOT:-$H/t3-run-batch-c721c1a2a-b1/uaf-repeat${UR_TAG:-}}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit8-batch-b2; bs=caplifive_vp2c_c721c1a2a; R=${UR_ROOT:-$H/t3-run-batch-c721c1a2a-b2/uaf-repeat${UR_TAG:-}}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "UR STOP: board $BRD" >> "$O"; return 2;;
  esac
  C=$H/t3-run-treeB/juliet-spatial/run/batch000/stage
  [ "$(sha256sum $C/images/controls.dom | cut -c1-16)" = b1decb0612856121 ] && [ "$(sha256sum $C/juliet-timeout | cut -c1-16)" = bb65b77ac802a9f3 ] \
    || { echo "UR STOP: controls.dom or juliet-timeout is not the Juliet run's" >> "$O"; return 1; }
  [ "$(git -C /tmp/capstone/wt-tools2 rev-parse --short=12 HEAD)" = e6c68a531557 ] && [ -z "$(git -C /tmp/capstone/wt-tools2 status --porcelain)" ] \
    || { echo "UR STOP: wt-tools2 is not a clean e6c68a531557" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 21fbfc397e7c7a42 ] \
    && [ "$(sha256sum $K/adapter/capstone-vexec | cut -c1-16)" = ${UR_VEXSHA:-b27afefc09c6cae6} ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } || { echo "UR STOP: kit $K" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "UR STOP: $R exists" >> "$O"; return 1; }
  ${UR_BUSY:-$BUSY} > /dev/null || { echo "UR STOP: board $BRD is busy" >> "$O"; return 1; }
  bid=$(${UR_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD ${UR_BOOKMIN:-75} "vp2 batch uaf-control repeat (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "UR STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "UR NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "UR BEGIN $(date +%T) board $BRD booking ${bid:-none} N=$N args ${UR_ARGS:-none} kit ${K##*/}" >> "$O"
  taskset -c 16-31 timeout 7200 ${UR_RUN:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/uaf-repeat.py} \
    --board-kit $K --tools /tmp/capstone/wt-tools2/capstone/bug-corpora/tools --controls $C/images/controls.dom --helper $C/juliet-timeout \
    --n $N ${UR_ARGS:-} --out $R > $R.log 2>&1; rc=$?
  [ -n "$bid" ] && ${UR_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "UR END rc=$rc $(date +%T)" >> "$O"
  grep -a "^KEEP \|^MAPPED-IMAGE\|^MAP \|^REFUSE-PTE\|^BOUNDS-0\|^RUN \|^TALLY\|^completed=\|^NO DATA\|^DMESG\|^  DM " $R.log | cut -c1-240 >> "$O"
  local tb=$R/work/board.txt c
  c="pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $tb 2>/dev/null) DRT1 $(grep -a -c '^DRT1:' $tb 2>/dev/null) FRTY $(grep -a -c 'FRTY:' $tb 2>/dev/null) Oops $(grep -a -c Oops $tb 2>/dev/null) INJS $(grep -a -c INJS $tb 2>/dev/null) INJL $(grep -a -c INJL $tb 2>/dev/null)"
  echo "UR readings: $c" >> "$O"
  echo "UR board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
