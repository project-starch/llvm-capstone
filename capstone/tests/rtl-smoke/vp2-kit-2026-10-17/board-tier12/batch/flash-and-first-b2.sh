#!/bin/bash
# Board 2: the batch bitstream c721c1a2a's flash and first boots (pre-registered in tier3-smoke/README.md; derived from
# flash-and-first-b1.sh by board-2 substitutions only). Waits for board 2's retire_verify arm to write its DONE line, then,
# each step gating the next:
#  1. the console store lists tree B's caplifive_vp2c_8de3c48cf.bit (the fallback); store-has.py's own control first:
#     a name that cannot exist must read missing (exit 1);
#  2. flash-bitstream.py (upload if unregistered, flash, power-cycle, read back), booked on the hub;
#  3. make-batch-kit.sh 1; 4. run-batch-first.sh 1 into a new directory.
# Usage: flash-and-first-b2.sh <log>. FB_* stub every external step for the dry run.
main() {
  set -u
  local O=${1:?log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local NAME=caplifive_vp2c_c721c1a2a BIT=/mnt/hdd/$USER/capstone/bitstreams/caplifive_vp2c_c721c1a2a.bit
  local SHA=8f4a486f2c9650412016e66d711af1ca8e0b003d2ac9a202a00f0d075b85ed6e OUT=${FB_OUT:-/mnt/hdd/$USER/capstone/t3-run-batch-c721c1a2a-b2}
  local WAITF=${FB_WAITF:-/tmp/capstone/treeB-session/pass/urrv-b2.out} t0 rc bid
  export FPGA_BOARD=2 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url-board2
  t0=$(date +%s)
  until grep -q "^UR board2 DONE\|^UR STOP" $WAITF 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${FB_WAITMAX:-5400} ] && { echo "FB STOP: board 2's arm did not end" >> $O; return 1; }
    sleep ${FB_SLEEP:-20}
  done
  echo "FB arm ended $(date +%T)" >> $O
  # Board 2 may be running another pre-registered arm by then: wait for it to be free (up to FB_BUSYMAX s), never interrupt it.
  t0=$(date +%s)
  until ${FB_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2} > /dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${FB_BUSYMAX:-14400} ] && { echo "FB STOP: board 2 stayed busy" >> $O; return 1; }
    sleep ${FB_SLEEP:-60}
  done
  echo "FB board 2 free $(date +%T)" >> $O
  ${FB_STORE:-python3 $S/store-has.py} no-such-bitstream-$$.bit >> $O 2>&1; rc=$?
  [ $rc = 1 ] || { echo "FB STOP: store-has control read rc=$rc, not 1" >> $O; return 1; }
  ${FB_STORE:-python3 $S/store-has.py} caplifive_vp2c_8de3c48cf.bit >> $O 2>&1 || { echo "FB STOP: the fallback is not registered" >> $O; return 1; }
  [ "$(sha256sum $BIT | cut -d' ' -f1)" = $SHA ] || { echo "FB STOP: local bitstream hash" >> $O; return 1; }
  bid=$(${FB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book 2 30 "vp2 batch c721c1a2a REFLASH (board 2, the lead's word)"); rc=$?
  [ $rc = 3 ] && { echo "FB STOP: the hub refused the flash booking (409): $bid" >> $O; return 1; }
  [ $rc = 0 ] || { echo "FB NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> $O; bid=; }
  echo "FB FLASH BEGIN $(date +%T) booking ${bid:-none}" >> $O
  ${FB_FLASH:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/drivers/flash-bitstream.py} $NAME.bit $BIT $SHA >> $O 2>&1; rc=$?
  [ -n "$bid" ] && ${FB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> $O 2>&1
  echo "FB FLASH END rc=$rc $(date +%T)" >> $O
  [ $rc = 0 ] || { echo "FB STOP: the flash did not read back $NAME.bit" >> $O; return 1; }
  ${FB_KIT:-bash $S/make-batch-kit.sh} 2 $NAME $O || { echo "FB STOP: kit" >> $O; return 1; }
  ${FB_FIRST:-bash $S/run-batch-first.sh} 2 $NAME $OUT; rc=$?
  echo "FB FIRST rc=$rc $(date +%T); SEQ:" >> $O; sed 's/^/  /' $OUT/SEQ >> $O 2>/dev/null
  echo "FB DONE $(date +%T)" >> $O
}
main "$@"; exit $?
