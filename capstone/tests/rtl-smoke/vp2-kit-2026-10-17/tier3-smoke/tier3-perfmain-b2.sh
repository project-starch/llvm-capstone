#!/bin/bash
# The perf evaluation's MAIN RUN, batch 1, on both boards (2026-10-26; pre-registered in README.md before any boot). The
# runtime lane's rule: try the four-kit split k1-k4 (perf/main1); if its firmware does not fit BOARD_FW_CAP, switch BOTH
# boards to the six-kit split s1-s6 (perf/main1b), and never mix the two. k1 has the largest stage, so it is baked first,
# alone, on board 1, and its firmware decides:
#   k1's firmware fits  -> board 1 runs k1 then k3, board 2 runs k2 then k4;
#   k1 stops on the cap -> board 1 runs s1, s3, s5 and board 2 runs s2, s4, s6;
#   anything else       -> stop, nothing else is launched.
# Each kit is one tier3-perfpilot-b2.sh boot (PP_SUB main1-<kit>, step 7200 s, booking 120 min, wrapper 32400 s).
# Usage: tier3-perfmain-b2.sh <log>. PM_RUNKIT stubs a kit's boot, PM_SEQ the k1 SEQ path and PM_PASS the logs' directory, for the dry run.
main() {
  set -u
  local O=${1:?log} S H=/mnt/hdd/$USER/capstone; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local M1=$H/perf/main1 M2=$H/perf/main1b
  declare -A SHA=([k1]=7d453b37a1ca5952 [k2]=0b8936b415b39679 [k3]=68a3b8587416b32a [k4]=8bca173a2ff4ed45
                  [s1]=018bea9b137c7730 [s2]=34d6374e904b43e3 [s3]=0067952dcb872fc6 [s4]=717dc6445d35f492
                  [s5]=2f290a53b7406955 [s6]=19286b13ccadd642)
  runkit() {   # runkit <board> <kit>: one boot; returns the wrapper's status
    local b=$1 k=$2 d; case $k in k*) d=$M1/$k;; s*) d=$M2/$k;; esac
    local K=${k^^} L=${PM_PASS:-/tmp/capstone/tree$([ $b = 1 ] && echo A || echo B)-session/pass}/perfmain-$k-b$b.out
    : > $L; echo "PM $k on board $b BEGIN $(date +%T)" >> "$O"
    if [ -n "${PM_RUNKIT:-}" ]; then $PM_RUNKIT $b $k $L; else
      PP_SUB=main1-$k PP_BOOKMIN=120 PP_WRAP_TO=32400 \
        PP_ARGS="--pilot-dir $d --runner run-$k.sh --runner-sha ${SHA[$k]} --done $K --step-s 7200" \
        bash $S/tier3-perfpilot-b2.sh $b $L; fi
    local rc=$?; echo "PM $k on board $b END rc=$rc $(date +%T) :: $(grep -a '^READ summary' $L | cut -c1-200)" >> "$O"
    grep -a "STOP" $L | sed 's/^/  /' >> "$O"; return $rc
  }
  seqline() { grep -a "^firmware \|^STOP" ${PM_SEQ:-$H/t3-run-b2-dcc6daf6e-b1/main1-k1/guest/SEQ} 2>/dev/null | head -1; }
  runkit 1 k1 & local p1=$!
  local t0=$(date +%s) s=
  until s=$(seqline); [ -n "$s" ] || ! kill -0 $p1 2>/dev/null; do
    [ $(( $(date +%s)-t0 )) -gt 3600 ] && { echo "PM STOP: no firmware decision from k1 in 1 h" >> "$O"; return 1; }; sleep 15; done
  s=$(seqline); echo "PM k1 firmware decision: ${s:-none}" >> "$O"
  case "$s" in
    "firmware "*)
      echo "PM SPLIT k1-k4: k1's firmware fits the cap" >> "$O"
      ( runkit 2 k2; runkit 2 k4 ) & local p2=$!
      wait $p1; runkit 1 k3; wait $p2;;
    *"over the cap"*)
      wait $p1; echo "PM SPLIT s1-s6: k1 stopped on the cap" >> "$O"
      ( runkit 2 s2; runkit 2 s4; runkit 2 s6 ) & local p2=$!
      runkit 1 s1; runkit 1 s3; runkit 1 s5; wait $p2;;
    *)
      wait $p1; echo "PM STOP: k1 gave no firmware decision (${s:-none}); nothing else launched" >> "$O"; return 1;;
  esac
  echo "PM DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
