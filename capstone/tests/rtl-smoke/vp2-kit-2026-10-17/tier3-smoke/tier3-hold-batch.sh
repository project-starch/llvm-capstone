#!/bin/bash
# live-peak `hold 100000` on one board (2026-10-23; pre-registered in README.md before the boot): the bitstream-2
# discriminator (a context at the module's DEFAULT capacity holding more than 65,535 live records) and its batch control.
# tier3-hold-board.py in one boardvm boot with the board's kit; the kit must carry no module parameters, and must be the
# board's batch kit (bkit8-batch-b<board>, module 21fbfc39) or its bitstream-2 kit (bkit9-b2-b<board>, module 215a09be).
# Starts when <after> reads <after ERE> ("-" at once). Usage: tier3-hold-batch.sh <board 1|2> <kit dir> <after|-> <ERE> <log>.
# HB_RUN / HB_BUSY / HB_BOOK / HB_ROOT stub the dry run.
main() {
  set -u
  local BRD=${1:?board} K=${2:?kit} W=${3:?after} WP=${4:?ERE} O=${5:?log} S end=$(( $(date +%s) + 72000 )) BUSY bid rc
  S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "HB STOP: board $BRD" >> "$O"; return 2;;
  esac
  if [ "$W" != - ]; then
    until grep -a -q -E "$WP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
    grep -a -q -E "$WP" "$W" || { echo "HB HOLD: $W did not read /$WP/" >> "$O"; return 1; }
    sleep 5
  fi
  local ms bs; ms=$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16); bs=$(sed -n 's/^BOARD_BITSTREAM=//p' $K/board.env)
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } \
    && case "${K##*/}:$ms" in bkit8-batch-b$BRD:21fbfc397e7c7a42|bkit9-b2-b$BRD:215a09be5a89360c) true;; *) false;; esac \
    || { echo "HB STOP: $K is not board $BRD's batch or bitstream-2 kit (module $ms)" >> "$O"; return 1; }
  [ -z "$(git -C /tmp/capstone/wt-newarms status --short --untracked-files=no)" ] && [ "$(git -C /tmp/capstone/wt-newarms rev-parse --short=12 HEAD)" = bcf5f1724110 ] \
    || { echo "HB STOP: wt-newarms is not a clean bcf5f1724110" >> "$O"; return 1; }
  local R=${HB_ROOT:-/mnt/hdd/$USER/capstone/t3-run-hold-${bs#caplifive_vp2c_}-b$BRD}
  [ -e $R ] && { echo "HB STOP: $R exists" >> "$O"; return 1; }
  ${HB_BUSY:-$BUSY} > /dev/null || { echo "HB STOP: board $BRD is busy" >> "$O"; return 1; }
  bid=$(${HB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 60 "vp2 hold 100000 on $bs (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "HB STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "HB NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "HB BEGIN $(date +%T) board $BRD bitstream $bs kit ${K##*/} module $ms booking ${bid:-none}" >> "$O"
  taskset -c 16-31 timeout 5400 ${HB_RUN:-python3 $S/tier3-hold-board.py} $K $R > $R.log 2>&1; rc=$?
  [ -n "$bid" ] && ${HB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "HB END rc=$rc $(date +%T)" >> "$O"
  grep -a "^HOLD boot\|^READ" $R.log | sed 's#/mnt/hdd/[^/]*#<mnt-hdd>#g' >> "$O"
  local B=$R/guest/board.txt
  [ -s "$B" ] && echo "HB readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) DRT1 $(grep -a -c '^DRT1:' $B) FRTY $(grep -a -c 'FRTY:' $B) FRTC $(grep -a -c 'FRTC' $B) Oops $(grep -a -c Oops $B) INJS $(grep -a -c INJS $B) INJL $(grep -a -c INJL $B)" >> "$O"
  echo "HB board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
