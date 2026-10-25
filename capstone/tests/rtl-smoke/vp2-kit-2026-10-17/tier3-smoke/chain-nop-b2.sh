#!/bin/bash
# The srz-nop module's own Tier 1 boot on BOARD 2 (2026-10-20; pre-registered in README.md before the boot): after
# q15 subset f on tree B ends (its out reads Q15 DONE, STOP or HOLD), run-tier1-nop-b2.sh t1-nop, then the gate: its four rung lines
# must equal t1-old's (/tmp/capstone/s26/run-t1-old-b2/SEQ). On PASS it writes CHAIN_PBNOP_GO, which starts the pb10 srz-nop pair
# (tier3-pb10-nop-b2.sh); on anything else it writes STOP, which holds it. Usage: chain-nop-b2.sh <q15bf out> <out>.
# Ends by itself. CZD_T1 stubs the boot for the dry run.
main() {
  set -u
  local W=${1:?q15bf out} O=${2:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local X=/tmp/capstone/s26 o n
  until grep -a -q "^Q15 DONE\|^Q15 STOP\|^Q15 HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^Q15 DONE\|^Q15 STOP\|^Q15 HOLD" "$W" 2>/dev/null || { echo "STOP: q15bf did not end" >> "$O"; return 1; }
  echo "CHAIN_NOP BEGIN $(date +%T)" >> "$O"
  ${CZD_T1:-bash $S/../board-tier12/srz/run-tier1-nop-b2.sh} t1-nop > "$O.t1" 2>&1; echo "CHAIN_NOP t1-nop END rc=$? $(date +%T)" >> "$O"
  o=$(grep -a '^RUNG ' $X/run-t1-old-b2/SEQ 2>/dev/null | sed 's/ oracle=.*//'); n=$(grep -a '^RUNG ' ${CZD_SEQ:-$X/run-t1-nop-b2/SEQ} 2>/dev/null | sed 's/ oracle=.*//')
  if [ -n "$n" ] && [ "$(echo "$n" | wc -l)" -eq 4 ] && [ "$o" = "$n" ]; then
    echo "CHAIN_NOP T1 GATE PASS: four rungs read as t1-old's" >> "$O"; echo "CHAIN_PBNOP_GO $(date +%T)" >> "$O"
  else echo "CHAIN_NOP T1 GATE FAIL: old [$o] nf [$n]" >> "$O"; echo "STOP: the srz-nop Tier 1 gate did not pass" >> "$O"; fi
  echo "CHAIN_NOP DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
