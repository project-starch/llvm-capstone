#!/bin/bash
# After sql0-fix (2026-10-19 evening; pre-registered in README.md before it runs). sql0-fix panicked as before with the
# S-23a strobe issued, so the stage-1 chain was stopped before its pb10 pair and this one replaces the rest, on board 1,
# each a cold boot, in order:
#   1. sql0-rf (tier3-sql-arm-rev2f.sh): sql0 on the refusal-read monitor 03a1ffe;
#   2. S-30's pair (run-s30.sh s30-old s30-new);
#   3. S-26's bracket (run-s26.sh q600 q300 q100), as pre-registered;
#   4. pb10-rf (tier3-pb10-rf.sh): a second instance of the refusal read.
# Usage: chain-rf.sh <out> [pid to wait for]. Ends by itself. CR_SQL / CR_S30 / CR_S26 / CR_PB stub the runners.
main() {
  set -u
  local O=${1:?out} P=${2:-} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); local B=$S/../board-tier12/s26 end=$(( $(date +%s) + 21600 ))
  while [ -n "$P" ] && kill -0 "$P" 2>/dev/null && [ "$(date +%s)" -lt $end ]; do sleep 30; done
  [ -n "$P" ] && kill -0 "$P" 2>/dev/null && { echo "CHAIN_RF HOLD: pid $P still running after 6 h" >> "$O"; return 1; }
  echo "CHAIN_RF BEGIN $(date +%T)" >> "$O"
  echo "CHAIN_RF_GO $(date +%T)" >> "$O"
  ${CR_SQL:-bash $S/tier3-sql-arm-rev2f.sh} "$O" "$O.sql0rf"; echo "CHAIN_RF SQL0RF END rc=$? $(date +%T) :: $(tail -1 "$O.sql0rf" 2>/dev/null)" >> "$O"
  ${CR_S30:-bash $B/run-s30.sh} s30-old s30-new > "$O.s30" 2>&1; echo "CHAIN_RF S30 END rc=$? $(date +%T)" >> "$O"
  ${CR_S26:-bash $B/run-s26.sh} q600 q300 q100 > "$O.bracket" 2>&1; echo "CHAIN_RF BRACKET END rc=$? $(date +%T)" >> "$O"
  echo "CHAIN_PB10RF_GO $(date +%T)" >> "$O"
  ${CR_PB:-bash $S/tier3-pb10-rf.sh} "$O" "$O.pb10rf"; echo "CHAIN_RF PB10RF END rc=$? $(date +%T) :: $(tail -1 "$O.pb10rf" 2>/dev/null)" >> "$O"
  echo "CHAIN_RF DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
