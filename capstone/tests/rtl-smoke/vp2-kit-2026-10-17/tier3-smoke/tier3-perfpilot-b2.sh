#!/bin/bash
# The perf evaluation's PILOT on bitstream 2 (2026-10-26; pre-registered in README.md before its boot): tier3-perfpilot.py
# on the perf kit bkit10-perf-b<board> = bkit9-b2-b<board> with only the adapter swapped to the runtime lane's bench build
# of dev's new module revision (launcher 978c09d61f774315, module 9c60fb53f80ce67e, lane runtime/bench-counters). The
# runtime lane's gate: modinfo must list parm bench_cycles, bench_instret and bench_entries. Output
# <mnt-hdd>/capstone/t3-run-b2-dcc6daf6e-b<board>/perfpilot. PP_BUSY / PP_BOOK / PP_RUN / PP_ROOT stub the dry run; PP_SUB names the output subdirectory and PP_ARGS passes tier3-perfpilot.py options (a later pilot or a main-run kit); PP_BOOKMIN and PP_WRAP_TO set the booking minutes and the wrapper timeout (defaults 75 and 14400).
# Usage: tier3-perfpilot-b2.sh <board 1|2> <out log>.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log} S H=/mnt/hdd/$USER/capstone K K0 R BUSY rc bid
  S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "PP STOP: board $BRD" >> "$O"; return 2;;
  esac
  K=$H/t3-smoke/bkit10-perf-b$BRD; K0=$H/t3-smoke/bkit9-b2-b$BRD; R=${PP_ROOT:-$H/t3-run-b2-dcc6daf6e-b$BRD/${PP_SUB:-perfpilot}}
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_dcc6daf6e" $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 9c60fb53f80ce67e ] && [ "$(sha256sum $K/adapter/capstone-vexec | cut -c1-16)" = 978c09d61f774315 ] \
    && [ ! -e $K/adapter/module/params ] \
    && diff -q <(grep -v "adapter/" $K/SHA256SUMS | sort) <(grep -v "adapter/" $K0/SHA256SUMS | sort) > /dev/null \
    || { echo "PP STOP: $K is not bkit9-b2-b$BRD with only the bench adapter swapped in" >> "$O"; return 1; }
  [ "$(modinfo -F parm $K/adapter/module/capstone_vm.ko | grep -c '^bench_\(cycles\|instret\|entries\):')" = 3 ] \
    || { echo "PP STOP: the module does not list the three bench parameters" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "PP STOP: $R exists" >> "$O"; return 1; }
  ${PP_BUSY:-$BUSY} > /dev/null || { echo "PP STOP: board $BRD is busy" >> "$O"; return 1; }
  bid=$(${PP_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD ${PP_BOOKMIN:-75} "vp2 bitstream 2 perf-eval ${PP_SUB:-perfpilot} (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "PP STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "PP NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "PP BEGIN $(date +%T) board $BRD kit ${K##*/} booking ${bid:-none}" >> "$O"
  taskset -c 16-31 timeout ${PP_WRAP_TO:-14400} ${PP_RUN:-python3 $S/tier3-perfpilot.py} --board-kit $K ${PP_ARGS:-} $R > $R.log 2>&1; rc=$?
  [ -n "$bid" ] && ${PP_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "PP END rc=$rc $(date +%T)" >> "$O"
  grep "^READ\|^PILOT" $R.log >> "$O"
  local B=$R/guest/board.txt
  [ -s "$B" ] && echo "PP readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) DRT1 $(grep -a -c '^DRT1:' $B) FRTY $(grep -a -c 'FRTY:' $B) Oops $(grep -a -c Oops $B) INJS $(grep -a -c INJS $B) INJL $(grep -a -c INJL $B)" >> "$O"
  echo "PP board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
