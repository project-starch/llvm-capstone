#!/bin/bash
# S-30's matched pair after stage 1 and the bracket (2026-10-19 evening): waits for CHAIN_FIX DONE in <stage-1 chain out>,
# then run-s30.sh s30-old s30-new on board 1. Usage: chain-s30.sh <stage-1 chain out> <out> [stage-1 pid]. Ends by itself:
# at DONE, when the stage-1 pid is gone without DONE (HOLD), or at a 24 h bound.
main() {
  set -u
  local W=${1:?stage-1 out} O=${2:?out} P=${3:-} S end=$(( $(date +%s) + 86400 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  until grep -a -q "^CHAIN_FIX DONE" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ] || { [ -n "$P" ] && ! kill -0 "$P" 2>/dev/null; }; do sleep 60; done
  grep -a -q "^CHAIN_FIX DONE" "$W" || { echo "CHAIN_S30 HOLD: stage 1 did not end" >> "$O"; return 1; }
  echo "CHAIN_S30 BEGIN $(date +%T)" >> "$O"
  ${CS_S30:-bash $S/run-s30.sh} s30-old s30-new >> "$O" 2>&1
  echo "CHAIN_S30 DONE rc=$? $(date +%T)" >> "$O"
}
main "$@"; exit $?
