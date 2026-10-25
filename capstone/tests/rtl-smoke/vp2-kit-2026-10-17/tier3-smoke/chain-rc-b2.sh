#!/bin/bash
# The srz-rc module's own Tier 1 boot on BOARD 2 (2026-10-20; pre-registered in README.md before the boot): after
# its go file reads RC_GO (board 2 free), run-tier1-rc-b2.sh t1-rc, then the gate: its four rung lines
# must equal t1-old's (/tmp/capstone/s26/run-t1-old-b2/SEQ). On PASS it writes CHAIN_PBRC_GO, which starts the pb10 srz-rc arm
# (tier3-pb10-rc-b2.sh); on anything else it writes STOP, which holds it. Usage: chain-rc-b2.sh <go file> <out>.
# Ends by itself. CZD_T1 stubs the boot for the dry run.
main() {
  set -u
  local W=${1:?go file} O=${2:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local X=/tmp/capstone/s26 o n
  until grep -a -q "^RC_GO" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^RC_GO" "$W" 2>/dev/null || { echo "STOP: no RC_GO" >> "$O"; return 1; }
  echo "CHAIN_RC BEGIN $(date +%T)" >> "$O"
  ${CZD_T1:-bash $S/../board-tier12/srz/run-tier1-rc-b2.sh} t1-rc > "$O.t1" 2>&1; echo "CHAIN_RC t1-rc END rc=$? $(date +%T)" >> "$O"
  o=$(grep -a '^RUNG ' $X/run-t1-old-b2/SEQ 2>/dev/null | sed 's/ oracle=.*//'); n=$(grep -a '^RUNG ' ${CZD_SEQ:-$X/run-t1-rc-b2/SEQ} 2>/dev/null | sed 's/ oracle=.*//')
  if [ -n "$n" ] && [ "$(echo "$n" | wc -l)" -eq 4 ] && [ "$o" = "$n" ]; then
    echo "CHAIN_RC T1 GATE PASS: four rungs read as t1-old's" >> "$O"; echo "CHAIN_PBRC_GO $(date +%T)" >> "$O"
  else echo "CHAIN_RC T1 GATE FAIL: old [$o] nf [$n]" >> "$O"; echo "STOP: the srz-rc Tier 1 gate did not pass" >> "$O"; fi
  echo "CHAIN_RC DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
