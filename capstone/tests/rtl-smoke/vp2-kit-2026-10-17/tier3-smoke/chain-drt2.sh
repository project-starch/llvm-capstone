#!/bin/bash
# sql0-drt2 after sql0-csr on board 1 (2026-10-19 evening; pre-registered in README.md before it runs): waits for chain-csr2's
# CHAIN_CSR2 DONE (or HOLD), then runs tier3-sql-arm-rev2j.sh. Usage: chain-drt2.sh <chain-csr2 out> <out>. Ends by itself
# (12 h bound). CD2_SQL stubs the runner.
main() {
  set -u
  local W=${1:?chain-csr2 out} O=${2:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  until grep -a -q "^CHAIN_CSR2 DONE\|^CHAIN_CSR2 HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  grep -a -q "^CHAIN_CSR2 DONE\|^CHAIN_CSR2 HOLD" "$W" || { echo "CHAIN_DRT2 HOLD: chain-csr2 did not end" >> "$O"; return 1; }
  echo "CHAIN_DRT2_GO $(date +%T)" >> "$O"
  ${CD2_SQL:-bash $S/tier3-sql-arm-rev2j.sh} "$O" "$O.sql0"; echo "CHAIN_DRT2 SQL0 END rc=$? $(date +%T)" >> "$O"
  echo "CHAIN_DRT2 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
