#!/bin/bash
# The perf evaluation's T3 arm on the other 88 programs, on BITSTREAM 3 (2026-10-27; pre-registered in README.md before
# any boot). The runtime lane's kits perf/main5/t3a1..t3a4 (pre-registered on runtime/perf-t3-all 92e59181f84d; 285, 285,
# 285 and 264 processes), one kit per boot, never split. Board 1 runs t3a1 then t3a3; board 2 runs t3a2 then t3a4.
# tier3-perfmain3-b2.sh with these changes: the main5 kits and their runner hashes; tier3-perfpilot-b3.sh (the bitstream-3
# perf kits bkit12-perf-b3-b<board>); step 10,800 s and booking 180 min, since these kits are 3.5x t3k1's 81 processes,
# which took 33 min. Before ANY board is booked, perfmain3-check.py must PASS on all four kits; one failure launches
# nothing. A kit runs only when the board's previous kit in this script ended rc=0.
# Usage: tier3-perfmain5-b3.sh <log>. PM5_RUNKIT stubs a kit's boot and PM5_PASS the logs' directory, for the dry run.
main() {
  set -u
  local O=${1:?log} S H=/mnt/hdd/$USER/capstone; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local M=${PM5_KITS:-$H/perf/main5}
  declare -A SHA=([t3a1]=95581eee5424dd8d [t3a2]=bd4ef4ee77c21fc6 [t3a3]=8d7f4b3cec3b260c [t3a4]=c2f7cc1b012f7c57)
  local k bad=0
  for k in t3a1 t3a2 t3a3 t3a4; do
    python3 $S/perfmain3-check.py $M/$k ${SHA[$k]} ${k^^} T1,T2,T3 >> "$O" 2>&1 || bad=1
  done
  [ $bad = 0 ] || { echo "PM5 STOP: a kit failed perfmain3-check.py; nothing launched" >> "$O"; return 1; }
  runkit() {   # runkit <board> <kit>: one boot; returns the wrapper's status
    local b=$1 k=$2 K=${2^^}
    local L=${PM5_PASS:-/tmp/capstone/tree$([ $b = 1 ] && echo A || echo B)-session/pass}/perfmain5-$k-b$b.out
    : > $L; echo "PM5 $k on board $b BEGIN $(date +%T)" >> "$O"
    if [ -n "${PM5_RUNKIT:-}" ]; then $PM5_RUNKIT $b $k $L; else
      PP_SUB=main5-$k PP_BOOKMIN=180 \
        PP_ARGS="--pilot-dir $M/$k --runner run-$k.sh --runner-sha ${SHA[$k]} --done $K --step-s 10800" \
        bash $S/tier3-perfpilot-b3.sh $b $L; fi
    local rc=$?; echo "PM5 $k on board $b END rc=$rc $(date +%T) :: $(grep -a '^READ summary' $L | cut -c1-200)" >> "$O"
    grep -a "STOP" $L | sed 's/^/  /' >> "$O"; return $rc
  }
  board() {   # board <board> <first kit> <second kit>
    runkit $1 $2 || { echo "PM5 board $1 HOLD: $2 ended rc!=0; $3 was not run" >> "$O"; return 1; }
    runkit $1 $3
  }
  board 2 t3a2 t3a4 & local p2=$!
  board 1 t3a1 t3a3
  wait $p2
  echo "PM5 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
