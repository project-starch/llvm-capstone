#!/bin/bash
# The perf evaluation's MAIN RUN, batch 2 (2026-10-26; pre-registered in README.md before its boot): the runtime lane's kit
# perf/main2/b2k1 (coremark, rv8-dhrystone, matmult-int; 24 processes) as a THIRD boot on whichever board ends its second
# batch-1 kit first: board 1's k3 or board 2's k4, read from tier3-perfmain-b2.sh's log. One tier3-perfpilot-b2.sh boot,
# PP_SUB main2-b2k1, step 2700 s, booking 45 min. Usage: tier3-perfmain2-b2.sh <batch-1 log> <log>. PM2_RUN stubs the boot.
main() {
  set -u
  local P=${1:?batch-1 log} O=${2:?log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local t0=$(date +%s) b=
  until b=$(grep -a -o "^PM k3 on board 1 END\|^PM k4 on board 2 END" "$P" | head -1 | grep -o "board [12]" | cut -c7); [ -n "$b" ]; do
    grep -q "^PM STOP" "$P" && { echo "PM2 STOP: batch 1 stopped" >> "$O"; return 1; }
    [ $(( $(date +%s)-t0 )) -gt 36000 ] && { echo "PM2 STOP: no second kit ended in 10 h" >> "$O"; return 1; }; sleep 30; done
  echo "PM2 b2k1 on board $b (its second batch-1 kit ended first) BEGIN $(date +%T)" >> "$O"
  local L=/tmp/capstone/tree$([ $b = 1 ] && echo A || echo B)-session/pass/perfmain-b2k1-b$b.out; : > $L
  if [ -n "${PM2_RUN:-}" ]; then $PM2_RUN $b $L; else
    PP_SUB=main2-b2k1 PP_BOOKMIN=45 \
      PP_ARGS="--pilot-dir /mnt/hdd/$USER/capstone/perf/main2/b2k1 --runner run-b2k1.sh --runner-sha e48f64aedd30000d --done B2K1 --step-s 2700" \
      bash $S/tier3-perfpilot-b2.sh $b $L; fi
  echo "PM2 END rc=$? $(date +%T) :: $(grep -a '^READ summary' $L | cut -c1-200)" >> "$O"
}
main "$@"; exit $?
