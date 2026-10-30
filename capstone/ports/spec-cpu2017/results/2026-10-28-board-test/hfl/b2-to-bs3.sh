#!/bin/bash
# b2-to-bs3.sh <log>: reflash board 2 to bitstream 3 (e7af48457, sha256 d6ed97b6..., the record in ISSUES.md and
# tier3-smoke/README.md) after b4-t1c, the same steps as b4-first.sh: board free, store-has control and fallback
# registered, local sha256, booked flash, readback by name. No boot. Reflashes need no confirmation (CLAUDE.md, 2026-10-30).
main() {
  set -u
  local O=${1:?log} H=/mnt/hdd/$USER/capstone NAME=caplifive_vp2c_e7af48457 BRD=2 rc bid
  local BIT=$H/bitstreams/$NAME.bit SHA=d6ed97b6d341ea058d497838e43558900823c36591c44046d3cfc0b7f55b423b
  local T=$H/wt-spec/capstone/tests/rtl-smoke BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2"
  export FPGA_BOARD=2 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url-board2
  grep -q '^TC DONE' $H/spec/b4-t1c.log || { echo "RB STOP: b4-t1c has not ended" >> $O; return 1; }
  $BUSY > /dev/null || { echo "RB STOP: board 2 busy" >> $O; return 1; }
  python3 $T/vp2-kit-2026-10-17/board-tier12/batch/store-has.py no-such-bitstream-$$.bit >> $O 2>&1; rc=$?
  [ $rc = 1 ] || { echo "RB STOP: store-has control rc=$rc, not 1" >> $O; return 1; }
  python3 $T/vp2-kit-2026-10-17/board-tier12/batch/store-has.py $NAME.bit >> $O 2>&1 || { echo "RB STOP: $NAME not registered" >> $O; return 1; }
  [ "$(sha256sum $BIT | cut -d' ' -f1)" = $SHA ] || { echo "RB STOP: local bitstream hash" >> $O; return 1; }
  bid=$(python3 /tmp/capstone/lab-hub/lab-book.py book $BRD 30 "vp2 bitstream 3 $NAME REFLASH (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "RB STOP: hub refused the flash booking: $bid" >> $O; return 1; }
  [ $rc = 0 ] || { echo "RB NOTE: booking failed rc=$rc; continuing unbooked" >> $O; bid=; }
  echo "RB FLASH BEGIN $(date +%T) board $BRD booking ${bid:-none}" >> $O
  python3 $T/drivers/flash-bitstream.py $NAME.bit $BIT $SHA >> $O 2>&1; rc=$?
  [ -n "$bid" ] && python3 /tmp/capstone/lab-hub/lab-book.py unbook $bid >> $O 2>&1
  echo "RB FLASH END rc=$rc $(date +%T)" >> $O
  [ $rc = 0 ] || { echo "RB STOP: flash rc=$rc" >> $O; return 1; }
  python3 $H/wt-spec/capstone/ports/spec-cpu2017/results/2026-10-28-board-test/b4/nv-readback.py $NAME.bit $SHA >> $O 2>&1; rc=$?
  case $rc in 0|3) echo "RB READBACK ok rc=$rc" >> $O;; *) echo "RB STOP: readback rc=$rc" >> $O; return 1;; esac
  echo "RB DONE $(date +%T)" >> $O
}
main "$@"; exit $?
