#!/bin/bash
# The runtime lane's bitstream-3 perf kit perf/main4/b3k1 (runner cbf60f75e5eb79ae; its pre-registration runtime/bench-counters
# 8bfc279da208) as ONE boot on board 2, after board 2's bitstream-3 chain (2026-10-27; pre-registered in README.md). It
# starts only when flash-and-first-bs3.sh's log reads "FB AFTER" and "FB DONE", i.e. the first boots read as predicted
# and the regression controls ran; "FB HOLD" or "FB STOP" starts nothing. Then perfmain3-check.py (arms T1,T2) must PASS,
# and tier3-perfpilot-b3.sh runs it: PP_SUB main4-b3k1, step 2700 s, booking 60 min.
# Usage: tier3-b3k1.sh <flash chain log> <log>. B3K_RUN stubs the boot; B3K_WAITMAX bounds the wait (default 12 h).
main() {
  set -u
  local W=${1:?flash chain log} O=${2:?log} S t0; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local K=/mnt/hdd/$USER/capstone/perf/main4/b3k1 SHA=cbf60f75e5eb79ae
  t0=$(date +%s)
  until grep -q "^FB DONE\|^FB STOP" $W 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${B3K_WAITMAX:-43200} ] && { echo "B3K STOP: the flash chain did not end" >> "$O"; return 1; }
    sleep ${B3K_SLEEP:-30}
  done
  grep -q "^FB AFTER rc=" $W && grep -q "^FB DONE" $W && ! grep -q "^FB HOLD\|^FB STOP" $W \
    || { echo "B3K HOLD: the flash chain did not reach its regression controls ($(grep '^FB HOLD\|^FB STOP' $W | head -1))" >> "$O"; return 1; }
  python3 $S/perfmain3-check.py $K $SHA B3K1 T1,T2 >> "$O" 2>&1 || { echo "B3K STOP: b3k1 failed perfmain3-check.py" >> "$O"; return 1; }
  echo "B3K BEGIN $(date +%T)" >> "$O"
  local L=/tmp/capstone/treeB-session/pass/perf-b3k1-b2.out; [ -n "${B3K_RUN:-}" ] && L=$O.pp; : > $L
  if [ -n "${B3K_RUN:-}" ]; then $B3K_RUN 2 $L; else
    PP_SUB=main4-b3k1 PP_BOOKMIN=60 \
      PP_ARGS="--pilot-dir $K --runner run-b3k1.sh --runner-sha $SHA --done B3K1 --step-s 2700" \
      bash $S/tier3-perfpilot-b3.sh 2 $L; fi
  echo "B3K END rc=$? $(date +%T) :: $(grep -a '^READ summary' $L | cut -c1-200)" >> "$O"
}
main "$@"; exit $?
