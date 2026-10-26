#!/bin/bash
# The perf evaluation's T3 arm on bitstream 2 (2026-10-27; pre-registered in README.md before any boot). The runtime lane's
# kits perf/main3b/t3k1..t3k3 (pre-registered on runtime/bench-counters edea56432aaa; 81, 60 and 60 processes), one kit per boot, never split. Board 1 runs
# t3k1 then t3k3; board 2 runs t3k2. Before ANY board is booked, perfmain3-check.py must PASS on all three kits (sums,
# runner hash, plan vs run lines, distinct T1/T2/T3 images, and every program's args equal to the main run's); one failure
# launches nothing. Each kit is one tier3-perfpilot-b2.sh boot: PP_SUB main3-<kit>, step 3600 s, booking 75 min.
# Usage: tier3-perfmain3-b2.sh <log>. PM3_RUNKIT stubs a kit's boot and PM3_PASS the logs' directory, for the dry run.
main() {
  set -u
  local O=${1:?log} S H=/mnt/hdd/$USER/capstone; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local M=$H/perf/main3b   # perf/main3 (first build) is VOID: huffbench-libc at "1 5"
  declare -A SHA=([t3k1]=0ecbbc7c4c9f5ec8 [t3k2]=fc24bc09d77351fc [t3k3]=37861b84fca1ac04)
  local k bad=0
  for k in t3k1 t3k2 t3k3; do
    python3 $S/perfmain3-check.py $M/$k ${SHA[$k]} ${k^^} >> "$O" 2>&1 || bad=1
  done
  [ $bad = 0 ] || { echo "PM3 STOP: a kit failed perfmain3-check.py; nothing launched" >> "$O"; return 1; }
  runkit() {   # runkit <board> <kit>: one boot; returns the wrapper's status
    local b=$1 k=$2 K=${2^^}
    local L=${PM3_PASS:-/tmp/capstone/tree$([ $b = 1 ] && echo A || echo B)-session/pass}/perfmain3-$k-b$b.out
    : > $L; echo "PM3 $k on board $b BEGIN $(date +%T)" >> "$O"
    if [ -n "${PM3_RUNKIT:-}" ]; then $PM3_RUNKIT $b $k $L; else
      PP_SUB=main3-$k PP_BOOKMIN=75 \
        PP_ARGS="--pilot-dir $M/$k --runner run-$k.sh --runner-sha ${SHA[$k]} --done $K --step-s 3600" \
        bash $S/tier3-perfpilot-b2.sh $b $L; fi
    local rc=$?; echo "PM3 $k on board $b END rc=$rc $(date +%T) :: $(grep -a '^READ summary' $L | cut -c1-200)" >> "$O"
    grep -a "STOP" $L | sed 's/^/  /' >> "$O"; return $rc
  }
  runkit 2 t3k2 & local p2=$!
  runkit 1 t3k1; runkit 1 t3k3
  wait $p2
  echo "PM3 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
