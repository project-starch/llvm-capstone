#!/bin/bash
# Q-34's store side on bitstream 2 (2026-10-26; pre-registered in README.md before its boots): tier3-misal.py on the corpus
# kit bkit9-b2-b<board> (module 215a09be, launcher fad465ce, no module parameters), one boot per board, output
# <mnt-hdd>/capstone/t3-run-b2-dcc6daf6e-b<board>/misal. MS_BUSY / MS_BOOK / MS_RUN stub the dry run.
# Usage: tier3-misal-b2.sh <board 1|2> <out log>.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log} S H=/mnt/hdd/$USER/capstone K R BUSY rc bid
  S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "MS STOP: board $BRD" >> "$O"; return 2;;
  esac
  K=$H/t3-smoke/bkit9-b2-b$BRD; R=${MS_ROOT:-$H/t3-run-b2-dcc6daf6e-b$BRD/misal}
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_dcc6daf6e" $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 215a09be5a89360c ] && [ "$(sha256sum $K/adapter/capstone-vexec | cut -c1-16)" = fad465ce4dea7121 ] \
    && [ ! -e $K/adapter/module/params ] \
    || { echo "MS STOP: $K is not board $BRD's bitstream-2 corpus kit" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "MS STOP: $R exists" >> "$O"; return 1; }
  ${MS_BUSY:-$BUSY} > /dev/null || { echo "MS STOP: board $BRD is busy" >> "$O"; return 1; }
  bid=$(${MS_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 30 "vp2 bitstream 2 Q-34 misaligned-store arm (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "MS STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "MS NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "MS BEGIN $(date +%T) board $BRD kit ${K##*/} booking ${bid:-none}" >> "$O"
  taskset -c 16-31 timeout 3600 ${MS_RUN:-python3 $S/tier3-misal.py} --board-kit $K $R > $R.log 2>&1; rc=$?
  [ -n "$bid" ] && ${MS_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "MS END rc=$rc $(date +%T)" >> "$O"
  grep "^READ\|^MISAL kit" $R.log >> "$O"
  local B=$R/guest/board.txt
  [ -s "$B" ] && echo "MS readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) DRT1 $(grep -a -c '^DRT1:' $B) FRTY $(grep -a -c 'FRTY:' $B) Oops $(grep -a -c Oops $B) INJS $(grep -a -c INJS $B) INJL $(grep -a -c INJL $B)" >> "$O"
  echo "MS board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
