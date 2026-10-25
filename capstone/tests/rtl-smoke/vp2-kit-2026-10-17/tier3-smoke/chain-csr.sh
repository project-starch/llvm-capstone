#!/bin/bash
# pb10-csr after cpython lt0 on board 1 (2026-10-19 evening; pre-registered in README.md before it runs). Waits for the cpython
# arm's out file to read CPY lt0 DONE or CPY STOP, then runs tier3-pb10-csr.sh. Usage: chain-csr.sh <cpy out> <out>.
# Ends by itself (12 h bound). CC_PB stubs the runner.
main() {
  set -u
  local W=${1:?cpy out} O=${2:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  until grep -a -q "^CPY lt0 DONE\|^CPY STOP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  grep -a -q "^CPY lt0 DONE\|^CPY STOP" "$W" || { echo "CHAIN_CSR HOLD: cpython did not end" >> "$O"; return 1; }
  echo "CHAIN_CSR_PB10_GO $(date +%T)" >> "$O"
  ${CC_PB:-bash $S/tier3-pb10-csr.sh} "$O" "$O.pb10"; echo "CHAIN_CSR PB10 END rc=$? $(date +%T) :: $(tail -1 "$O.pb10" 2>/dev/null)" >> "$O"
  echo "CHAIN_CSR DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
