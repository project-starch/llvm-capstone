#!/bin/bash
# After S-30's pair (2026-10-19 evening; pre-registered in README.md before it runs). chain-rf was stopped during the S-30
# pair; this runs the rest on board 1, each a cold boot, in order:
#   1. sql0-drt (tier3-sql-arm-rev2g.sh): sql0 on the data-retry monitor f13dbc5;
#   2. pb10-drt (tier3-pb10-drt.sh): pb10 on the same monitor (pb10-rf withdrawn: this prints the same refusal read);
#   3. S-26's bracket (run-s26.sh q600 q300 q100), as pre-registered.
# Usage: chain-drt.sh <out> [pid to wait for]. Ends by itself. CD_SQL / CD_PB / CD_S26 stub the runners.
main() {
  set -u
  local O=${1:?out} P=${2:-} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); local B=$S/../board-tier12/s26 end=$(( $(date +%s) + 21600 ))
  while [ -n "$P" ] && kill -0 "$P" 2>/dev/null && [ "$(date +%s)" -lt $end ]; do sleep 30; done
  [ -n "$P" ] && kill -0 "$P" 2>/dev/null && { echo "CHAIN_DRT HOLD: pid $P still running after 6 h" >> "$O"; return 1; }
  echo "CHAIN_DRT BEGIN $(date +%T)" >> "$O"
  echo "CHAIN_DRT_SQL_GO $(date +%T)" >> "$O"
  ${CD_SQL:-bash $S/tier3-sql-arm-rev2g.sh} "$O" "$O.sql0drt"; echo "CHAIN_DRT SQL0 END rc=$? $(date +%T) :: $(tail -1 "$O.sql0drt" 2>/dev/null)" >> "$O"
  echo "CHAIN_DRT_PB10_GO $(date +%T)" >> "$O"
  ${CD_PB:-bash $S/tier3-pb10-drt.sh} "$O" "$O.pb10drt"; echo "CHAIN_DRT PB10 END rc=$? $(date +%T) :: $(tail -1 "$O.pb10drt" 2>/dev/null)" >> "$O"
  ${CD_S26:-bash $B/run-s26.sh} q600 q300 q100 > "$O.bracket" 2>&1; echo "CHAIN_DRT BRACKET END rc=$? $(date +%T)" >> "$O"
  echo "CHAIN_DRT DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
