#!/bin/bash
# Board 2: BITSTREAM 2's (18-bit index) flash and first boots, on the lead's word of 2026-10-23 ("reflash board 2 after
# bitstream 2 synthesis finish"; pre-registered in tier3-smoke/README.md). flash-and-first-b2.sh with: the bitstream's
# name, local file and sealed sha256 as arguments (known only at the seal); the wait is for board 2's sql0 run to end
# (any PG DONE/STOP/VOID/MUST-FIRE line), then for the board to be free; the fallback that must be registered is the
# batch bitstream caplifive_vp2c_c721c1a2a.bit (the board's current one); make-b2-kit.sh and run-b2-first.sh in place of
# the batch's. Each step gates the next. Usage: flash-and-first-bs2-b2.sh <log> <bitstream name, no .bit> <local .bit>
# <sha256>. FB_* stub every external step for the dry run.
main() {
  set -u
  local O=${1:?log} NAME=${2:?bitstream name} BIT=${3:?local .bit} SHA=${4:?sha256} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  case $NAME in caplifive_*) ;; *) echo "FB STOP: bitstream name $NAME" >> $O; return 2;; esac
  [ ${#SHA} = 64 ] || { echo "FB STOP: sha256 $SHA is not 64 hex digits" >> $O; return 2; }
  local OUT=${FB_OUT:-/mnt/hdd/$USER/capstone/t3-run-b2-${NAME#caplifive_vp2c_}-b2}
  local WAITF=${FB_WAITF:-/tmp/capstone/treeB-session/pass/sql0-b2.out} t0 rc bid
  export FPGA_BOARD=2 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url-board2
  t0=$(date +%s)
  until grep -q "^PG DONE\|^PG STOP\|^PG VOID\|^PG MUST-FIRE FAIL" $WAITF 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${FB_WAITMAX:-36000} ] && { echo "FB STOP: board 2's sql0 run did not end" >> $O; return 1; }
    sleep ${FB_SLEEP:-20}
  done
  echo "FB sql0 ended $(date +%T)" >> $O
  # Board 2 may be running another pre-registered arm by then: wait for it to be free (up to FB_BUSYMAX s), never interrupt it.
  t0=$(date +%s)
  until ${FB_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2} > /dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${FB_BUSYMAX:-14400} ] && { echo "FB STOP: board 2 stayed busy" >> $O; return 1; }
    sleep ${FB_SLEEP:-60}
  done
  echo "FB board 2 free $(date +%T)" >> $O
  ${FB_STORE:-python3 $S/store-has.py} no-such-bitstream-$$.bit >> $O 2>&1; rc=$?
  [ $rc = 1 ] || { echo "FB STOP: store-has control read rc=$rc, not 1" >> $O; return 1; }
  ${FB_STORE:-python3 $S/store-has.py} caplifive_vp2c_c721c1a2a.bit >> $O 2>&1 || { echo "FB STOP: the fallback (the batch bitstream) is not registered" >> $O; return 1; }
  [ "$(sha256sum $BIT | cut -d' ' -f1)" = $SHA ] || { echo "FB STOP: local bitstream hash" >> $O; return 1; }
  bid=$(${FB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book 2 30 "vp2 bitstream 2 $NAME REFLASH (board 2, the lead's word)"); rc=$?
  [ $rc = 3 ] && { echo "FB STOP: the hub refused the flash booking (409): $bid" >> $O; return 1; }
  [ $rc = 0 ] || { echo "FB NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> $O; bid=; }
  echo "FB FLASH BEGIN $(date +%T) booking ${bid:-none}" >> $O
  ${FB_FLASH:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/drivers/flash-bitstream.py} $NAME.bit $BIT $SHA >> $O 2>&1; rc=$?
  [ -n "$bid" ] && ${FB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> $O 2>&1
  echo "FB FLASH END rc=$rc $(date +%T)" >> $O
  [ $rc = 0 ] || { echo "FB STOP: the flash did not read back $NAME.bit" >> $O; return 1; }
  ${FB_KIT:-bash $S/make-b2-kit.sh} 2 $NAME $O || { echo "FB STOP: kit" >> $O; return 1; }
  grep -q "^MK module 215a09be5a89360c launcher fad465ce4dea7121 DONE" $O || { echo "FB STOP: make-b2-kit did not end DONE" >> $O; return 1; }
  ${FB_FIRST:-bash $S/run-b2-first.sh} 2 $NAME $OUT; rc=$?
  echo "FB FIRST rc=$rc $(date +%T); SEQ:" >> $O; sed 's/^/  /' $OUT/SEQ >> $O 2>/dev/null
  echo "FB DONE $(date +%T)" >> $O
}
main "$@"; exit $?
