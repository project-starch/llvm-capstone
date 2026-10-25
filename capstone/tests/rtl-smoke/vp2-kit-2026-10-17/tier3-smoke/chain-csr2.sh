#!/bin/bash
# sql0-csr after pb10-csr on board 1 (2026-10-19 evening; pre-registered in README.md before it runs): waits for chain-csr.sh's
# CHAIN_CSR DONE (or HOLD), then runs tier3-sql-arm-rev2i.sh. Usage: chain-csr2.sh <chain-csr out> <out>. Ends by itself
# (12 h bound). CC_SQL stubs the runner.
main() {
  set -u
  local W=${1:?chain-csr out} O=${2:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  until grep -a -q "^CHAIN_CSR DONE\|^CHAIN_CSR HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  grep -a -q "^CHAIN_CSR DONE\|^CHAIN_CSR HOLD" "$W" || { echo "CHAIN_CSR2 HOLD: chain-csr did not end" >> "$O"; return 1; }
  echo "CHAIN_CSR_SQL_GO $(date +%T)" >> "$O"
  ${CC_SQL:-bash $S/tier3-sql-arm-rev2i.sh} "$O" "$O.sql0"; echo "CHAIN_CSR2 SQL0 END rc=$? $(date +%T)" >> "$O"
  echo "CHAIN_CSR2 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
