#!/bin/bash
# Board 1's tail of the perf main run, batch 1, after the runtime lane inserted the FP discriminator between k1 and k3
# (2026-10-26; pre-registered in README.md before its boots). tier3-perfmain-b2.sh's main shell was stopped before k1
# ended, so it never launches k3; its k1 and k2->k4 subshells run on and still write their PM lines to its log. This
# waits for "PM k1 on board 1 END" in that log, then runs on board 1:
#   1. the FP discriminator: tier3-fpdiag.py through tier3-perfpilot-b2.sh (PP_SUB fpdiag, booking 30 min);
#   2. k3, exactly as the coordinator's runkit would (PP_SUB main1-k3, step 7200 s, booking 120 min, wrapper 32400 s),
#      with its PM BEGIN/END lines in the same log, which tier3-perfmain2-b2.sh (batch 2's placement) reads.
# Usage: tier3-perfmain-b1tail.sh <batch-1 log>. T_RUN stubs both boots for the dry run.
main() {
  set -u
  local O=${1:?batch-1 log} S H=/mnt/hdd/$USER/capstone P=/tmp/capstone/treeA-session/pass; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local t0=$(date +%s)
  until grep -q "^PM k1 on board 1 END" "$O"; do
    [ $(( $(date +%s)-t0 )) -gt 14400 ] && { echo "PM STOP: board 1's k1 did not end in 4 h" >> "$O"; return 1; }; sleep 30; done
  local L=${T_PASS:-$P}/fpdiag-b1.out; : > $L; echo "PM fpdiag on board 1 BEGIN $(date +%T)" >> "$O"
  if [ -n "${T_RUN:-}" ]; then $T_RUN fpdiag $L; else
    PP_SUB=fpdiag PP_BOOKMIN=30 PP_RUN="python3 $S/tier3-fpdiag.py" bash $S/tier3-perfpilot-b2.sh 1 $L; fi
  echo "PM fpdiag on board 1 END rc=$? $(date +%T)" >> "$O"
  L=${T_PASS:-$P}/perfmain-k3-b1.out; : > $L; echo "PM k3 on board 1 BEGIN $(date +%T)" >> "$O"
  if [ -n "${T_RUN:-}" ]; then $T_RUN k3 $L; else
    PP_SUB=main1-k3 PP_BOOKMIN=120 PP_WRAP_TO=32400 \
      PP_ARGS="--pilot-dir $H/perf/main1/k3 --runner run-k3.sh --runner-sha 68a3b8587416b32a --done K3 --step-s 7200" \
      bash $S/tier3-perfpilot-b2.sh 1 $L; fi
  local rc=$?; echo "PM k3 on board 1 END rc=$rc $(date +%T) :: $(grep -a '^READ summary' $L | cut -c1-200)" >> "$O"
  grep -a "STOP" $L | sed 's/^/  /' >> "$O"
}
main "$@"; exit $?
