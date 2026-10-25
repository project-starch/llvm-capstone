#!/bin/bash
# The S-23a/S-26 monitor revision's stage 1, then S-26's silicon bracket, on board 1, tree A (2026-10-19 evening;
# pre-registered in README.md before any of it runs). In order, each a cold boot:
#   1. fix1, the monitor's canary (run-s26.sh fix1: b0-stats, vm-contract, b0-hello). PASS writes CHAIN_FIX1_PASS.
#   2. sql0-fix (tier3-sql-arm-rev2e.sh), only on CHAIN_FIX1_PASS.
#   3. pb10's matched pair (tier3-pb10-pair.sh: pb10-fix, then pb10-treeA), only on CHAIN_FIX1_PASS.
#   4. S-26's bracket (run-s26.sh q600 q300 q100), on the OLD pair monitor, so it runs whatever the canary read.
# Usage: chain-fix-stage1.sh <out>. Ends by itself. CF_S26 / CF_SQL / CF_PB10 stub the three runners for the dry run.
main() {
  set -u
  local O=${1:?out} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local B=$S/../board-tier12/s26
  local S26=${CF_S26:-bash $B/run-s26.sh} SQL=${CF_SQL:-bash $S/tier3-sql-arm-rev2e.sh} PB=${CF_PB10:-bash $S/tier3-pb10-pair.sh}
  echo "CHAIN_FIX BEGIN $(date +%T)" >> "$O"
  $S26 fix1 > "$O.fix1" 2>&1
  if grep -a -q "^S26 fix1 JUDGE rc=0 " "$O.fix1"; then
    echo "CHAIN_FIX1_PASS $(date +%T)" >> "$O"
    $SQL "$O" "$O.sql0" ; echo "CHAIN_SQL0 END rc=$? $(date +%T) :: $(tail -1 "$O.sql0" 2>/dev/null)" >> "$O"
    echo "CHAIN_PB10_GO $(date +%T)" >> "$O"
    $PB "$O" "$O.pb10"; echo "CHAIN_PB10 END rc=$? $(date +%T) :: $(tail -1 "$O.pb10" 2>/dev/null)" >> "$O"
  else
    echo "CHAIN_FIX1_FAIL $(date +%T) :: $(grep -a '^S26 fix1 JUDGE\|STOP' "$O.fix1" | tail -1); sql0 and pb10 do not run" >> "$O"
  fi
  $S26 q600 q300 q100 > "$O.bracket" 2>&1; echo "CHAIN_BRACKET END rc=$? $(date +%T)" >> "$O"
  echo "CHAIN_FIX DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
