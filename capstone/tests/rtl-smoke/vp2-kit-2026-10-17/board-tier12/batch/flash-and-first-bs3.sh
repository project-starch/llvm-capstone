#!/bin/bash
# BITSTREAM 3's (capstone-ariane e7af48457: R-62/R-63 divider fixes) flash and first boots on ONE board, on the lead's own
# word in the board lane's session (2026-10-27: "as soon as the bitstream is ready, do the re-flash; you have my
# go-ahead now"; pre-registered in tier3-smoke/README.md). flash-and-first-bs2-b<board>.sh with these changes:
# - the wait is for that board's T3 perf kits to end (tier3-perfmain3-b2.sh's log: board 2 after t3k2, board 1 after
#   t3k3, or the run's STOP/end), then for the board to be free;
# - the fallback that must be registered is bitstream 2, caplifive_vp2c_dcc6daf6e.bit;
# - the kits are make-b3-kit.sh copies of the board's bitstream-2 kits (bkit9-b2 -> bkit11-b3, bkit10-perf ->
#   bkit12-perf-b3), changing only BOARD_BITSTREAM;
# - the first boots are run-b3-first.sh.
# Each step gates the next.
# Usage: flash-and-first-bs3.sh <board 1|2> <log> <bitstream name, no .bit> <local .bit> <sha256>.
# FB_* stub every external step for the dry run.
main() {
  set -u
  local BRD=${1:?board} O=${2:?log} NAME=${3:?bitstream name} BIT=${4:?local .bit} SHA=${5:?sha256}
  local S H=/mnt/hdd/$USER/capstone; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  case $NAME in caplifive_*e7af48457*) ;; *) echo "FB STOP: bitstream name $NAME is not an e7af48457 build" >> $O; return 2;; esac
  [ ${#SHA} = 64 ] || { echo "FB STOP: sha256 $SHA is not 64 hex digits" >> $O; return 2; }
  local OUT=${FB_OUT:-$H/t3-run-b3-e7af48457-b$BRD} WAITF=${FB_WAITF:-/tmp/capstone/treeA-session/pass/perfmain3.log} BUSY pat t0 rc bid
  case $BRD in
    1) export FPGA_BOARD=1; unset CAPSTONE_FPGA_URL_FILE; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1"
       pat="^PM3 t3k3 on board 1 END";;
    2) export FPGA_BOARD=2 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url-board2
       BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2"; pat="^PM3 t3k2 on board 2 END";;
    *) echo "FB STOP: board $BRD" >> $O; return 2;;
  esac
  t0=$(date +%s)
  until grep -q "$pat\|^PM3 STOP\|^ORCH rc=" $WAITF 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${FB_WAITMAX:-21600} ] && { echo "FB STOP: board $BRD's T3 kits did not end" >> $O; return 1; }
    sleep ${FB_SLEEP:-30}
  done
  echo "FB board $BRD's T3 kits ended $(date +%T): $(grep "$pat\|^PM3 STOP\|^ORCH rc=" $WAITF | head -1 | cut -c1-120)" >> $O
  t0=$(date +%s)
  until ${FB_BUSY:-$BUSY} > /dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${FB_BUSYMAX:-14400} ] && { echo "FB STOP: board $BRD stayed busy" >> $O; return 1; }
    sleep ${FB_SLEEP:-60}
  done
  echo "FB board $BRD free $(date +%T)" >> $O
  ${FB_STORE:-python3 $S/store-has.py} no-such-bitstream-$$.bit >> $O 2>&1; rc=$?
  [ $rc = 1 ] || { echo "FB STOP: store-has control read rc=$rc, not 1" >> $O; return 1; }
  ${FB_STORE:-python3 $S/store-has.py} caplifive_vp2c_dcc6daf6e.bit >> $O 2>&1 || { echo "FB STOP: the fallback (bitstream 2) is not registered" >> $O; return 1; }
  [ "$(sha256sum $BIT | cut -d' ' -f1)" = $SHA ] || { echo "FB STOP: local bitstream hash" >> $O; return 1; }
  bid=$(${FB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 30 "vp2 bitstream 3 $NAME REFLASH (board $BRD, the lead's word)"); rc=$?
  [ $rc = 3 ] && { echo "FB STOP: the hub refused the flash booking (409): $bid" >> $O; return 1; }
  [ $rc = 0 ] || { echo "FB NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> $O; bid=; }
  echo "FB FLASH BEGIN $(date +%T) board $BRD booking ${bid:-none}" >> $O
  ${FB_FLASH:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/drivers/flash-bitstream.py} $NAME.bit $BIT $SHA >> $O 2>&1; rc=$?
  [ -n "$bid" ] && ${FB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> $O 2>&1
  echo "FB FLASH END rc=$rc $(date +%T)" >> $O
  [ $rc = 0 ] || { echo "FB STOP: the flash did not read back $NAME.bit" >> $O; return 1; }
  ${FB_KIT:-bash $S/make-b3-kit.sh} ${FB_KSRC:-$H/t3-smoke}/bkit9-b2-b$BRD ${FB_KDST:-$H/t3-smoke}/bkit11-b3-b$BRD $NAME $O \
    && ${FB_KIT:-bash $S/make-b3-kit.sh} ${FB_KSRC:-$H/t3-smoke}/bkit10-perf-b$BRD ${FB_KDST:-$H/t3-smoke}/bkit12-perf-b3-b$BRD $NAME $O \
    || { echo "FB STOP: kit" >> $O; return 1; }
  ${FB_FIRST:-bash $S/run-b3-first.sh} $BRD $NAME $OUT; rc=$?
  echo "FB FIRST rc=$rc $(date +%T); SEQ:" >> $O; sed 's/^/  /' $OUT/SEQ >> $O 2>/dev/null
  echo "FB DONE $(date +%T)" >> $O
}
main "$@"; exit $?
