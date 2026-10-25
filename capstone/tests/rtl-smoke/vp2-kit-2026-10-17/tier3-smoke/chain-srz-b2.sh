#!/bin/bash
# The srz module on BOARD 2, tree B (2026-10-20; pre-registered in README.md before the boots), after sql0-csr-treeB ends:
#   1. t1-old, then t1-srz (board-tier12/srz/run-tier1-srz-b2.sh): rev7's Tier 1 boot on the retry monitor line, differing
#      only in capstone_vm.ko (39d8492d vs 607164b9);
#   2. sql0-srz-treeB (tier3-sql-arm-rev2l.sh), ONLY if t1-srz's four rungs read as t1-old's (the new module's own Tier 1
#      gate); otherwise CHAIN_SRZ HOLD.
# Usage: chain-srz-b2.sh <b2c out> <out>. Ends by itself (24 h bound). CZ_T1 / CZ_SQL stub the runners.
main() {
  set -u
  local W=${1:?b2c out} O=${2:?out} S end=$(( $(date +%s) + 86400 )) a; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local T1=${CZ_T1:-bash $S/../board-tier12/srz/run-tier1-srz-b2.sh} X=${CZ_X:-/tmp/capstone/s26}
  until grep -a -q "^SQL DONE\|^SQL STOP\|^SQL HOLD\|^SQL VOID" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  echo "CHAIN_SRZ BEGIN $(date +%T)" >> "$O"
  for a in t1-old t1-srz; do
    $T1 $a > "$O.$a" 2>&1; echo "CHAIN_SRZ $a END rc=$? $(date +%T) :: $(grep -a '^RUNG ' $X/run-$a-b2/SEQ 2>/dev/null | tr '\n' ' ')" >> "$O"
  done
  local o n; o=$(grep -a '^RUNG ' $X/run-t1-old-b2/SEQ 2>/dev/null | sed 's/ oracle=.*//'); n=$(grep -a '^RUNG ' $X/run-t1-srz-b2/SEQ 2>/dev/null | sed 's/ oracle=.*//')
  if [ -n "$n" ] && [ "$(echo "$n" | wc -l)" -eq 4 ] && [ "$o" = "$n" ]; then
    echo "CHAIN_SRZ T1 GATE PASS: t1-srz's four rungs read as t1-old's" >> "$O"
    echo "CHAIN_SRZ_SQL_GO $(date +%T)" >> "$O"
    ${CZ_SQL:-bash $S/tier3-sql-arm-rev2l.sh} "$O" "$O.sql0"; echo "CHAIN_SRZ SQL0 END rc=$? $(date +%T)" >> "$O"
  else
    echo "CHAIN_SRZ HOLD: t1-srz did not read as t1-old (old [$o] srz [$n]); sql0-srz-treeB does not run" >> "$O"
  fi
  echo "CHAIN_SRZ DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
