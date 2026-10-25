#!/bin/bash
# Tree B's discriminator on BOARD 2 (2026-10-19 evening; pre-registered in README.md before it runs): after S-29's seven bare
# arms on board 2 (tier3-sv-batch-treeB2.sh) end, sql0 then pb10 on tree B with the data-retry monitor f13dbc5, each a cold
# boot. Usage: chain-treeB2-linux.sh <sv-batch out> <out>. Ends by itself. CB_SQL / CB_PB stub the runners.
main() {
  set -u
  local W=${1:?sv-batch out} O=${2:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  until grep -a -q "^SVBATCH \(DONE\|STOP\|HOLD\)" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^SVBATCH DONE" "$W" || { echo "CHAIN_B2 HOLD: the S-29 batch did not end DONE ($(grep -a '^SVBATCH' "$W" | tail -1))" >> "$O"; return 1; }
  echo "CHAIN_B2 BEGIN $(date +%T)" >> "$O"; echo "CHAIN_B2_SQL_GO $(date +%T)" >> "$O"
  ${CB_SQL:-bash $S/tier3-sql-arm-rev2h.sh} "$O" "$O.sql0"; echo "CHAIN_B2 SQL0 END rc=$? $(date +%T) :: $(tail -1 "$O.sql0" 2>/dev/null)" >> "$O"
  echo "CHAIN_B2_PB10_GO $(date +%T)" >> "$O"
  ${CB_PB:-bash $S/tier3-pb10-treeB2.sh} "$O" "$O.pb10"; echo "CHAIN_B2 PB10 END rc=$? $(date +%T) :: $(tail -1 "$O.pb10" 2>/dev/null)" >> "$O"
  echo "CHAIN_B2 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
