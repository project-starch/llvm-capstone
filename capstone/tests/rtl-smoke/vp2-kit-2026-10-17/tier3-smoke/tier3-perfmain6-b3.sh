#!/bin/bash
# The 15 allocating programs' T3 kits on BITSTREAM 3 (2026-10-28; the runtime lane's pre-registration is
# runtime/perf-t3-all 811dc7a43eb3, this runner's is in README.md, both before any boot). The kits perf/main3b/t3k1..t3k3,
# unchanged from the bitstream-2 T3 run (81, 60 and 60 processes), one kit per boot, never split. Board 1 runs t3k1 then
# t3k3; board 2 runs t3k2. tier3-perfmain3-b2.sh with these changes: tier3-perfpilot-b3.sh (bkit12-perf-b3-b<board>),
# output main6-<kit>, and t3k3 runs only when t3k1 ended rc=0, as in tier3-perfmain5-b3.sh. Each board waits until its
# corpus arm in tier3-corpus-b3.sh has ended (board 1: lt1, board 2: lt0), read from the corpus log given as $2; a board
# whose arm has not ended within 14 h launches nothing. Before ANY board is booked, perfmain3-check.py must PASS on all
# three kits; one failure launches nothing.
# Usage: tier3-perfmain6-b3.sh <log> <corpus log>. PM6_RUNKIT stubs a kit's boot, PM6_PASS the logs' directory,
# PM6_KITS the kits' directory and PM6_POLL / PM6_WAITMAX the wait, for the dry run.
main() {
  set -u
  local O=${1:?log} C=${2:?corpus log} S H=/mnt/hdd/$USER/capstone; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local M=${PM6_KITS:-$H/perf/main3b}
  declare -A SHA=([t3k1]=0ecbbc7c4c9f5ec8 [t3k2]=fc24bc09d77351fc [t3k3]=37861b84fca1ac04)
  local k bad=0
  for k in t3k1 t3k2 t3k3; do
    python3 $S/perfmain3-check.py $M/$k ${SHA[$k]} ${k^^} >> "$O" 2>&1 || bad=1
  done
  [ $bad = 0 ] || { echo "PM6 STOP: a kit failed perfmain3-check.py; nothing launched" >> "$O"; return 1; }
  waitfree() {   # waitfree <board> <corpus tag>: returns 0 once that arm's END line is in the corpus log
    local b=$1 t=$2 t0; t0=$(date +%s)
    echo "PM6 board $b WAIT for corpus $t END $(date +%T)" >> "$O"
    until grep -q "^CO $t on board $b END rc=" "$C" 2>/dev/null; do
      [ $(( $(date +%s) - t0 )) -gt ${PM6_WAITMAX:-50400} ] && {
        echo "PM6 board $b STOP: corpus $t did not end within ${PM6_WAITMAX:-50400} s; nothing launched" >> "$O"; return 1; }
      sleep ${PM6_POLL:-60}
    done
    echo "PM6 board $b FREE: $(grep "^CO $t on board $b END rc=" "$C" | cut -c1-120)" >> "$O"
  }
  runkit() {   # runkit <board> <kit>: one boot; returns the wrapper's status
    local b=$1 k=$2 K=${2^^}
    local L=${PM6_PASS:-/tmp/capstone/tree$([ $b = 1 ] && echo A || echo B)-session/pass}/perfmain6-$k-b$b.out
    : > $L; echo "PM6 $k on board $b BEGIN $(date +%T)" >> "$O"
    if [ -n "${PM6_RUNKIT:-}" ]; then $PM6_RUNKIT $b $k $L; else
      PP_SUB=main6-$k PP_BOOKMIN=75 \
        PP_ARGS="--pilot-dir $M/$k --runner run-$k.sh --runner-sha ${SHA[$k]} --done $K --step-s 3600" \
        bash $S/tier3-perfpilot-b3.sh $b $L; fi
    local rc=$?; echo "PM6 $k on board $b END rc=$rc $(date +%T) :: $(grep -a '^READ summary' $L | cut -c1-200)" >> "$O"
    grep -a "STOP" $L | sed 's/^/  /' >> "$O"; return $rc
  }
  { waitfree 2 lt0 && runkit 2 t3k2; } & local p2=$!
  if waitfree 1 lt1; then
    if runkit 1 t3k1; then runkit 1 t3k3; else echo "PM6 board 1 HOLD: t3k1 ended rc!=0; t3k3 was not run" >> "$O"; fi
  fi
  wait $p2
  echo "PM6 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
