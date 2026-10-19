#!/bin/bash
# The node-floor module's own Tier 1 boot on BOARD 2 (2026-10-20; pre-registered in README.md before the boot): after
# sql0-frs-treeB ends (its out reads SQL DONE, STOP or HOLD), run-tier1-nf-b2.sh t1-nf, then the gate: its four rung lines
# must equal t1-old's (/tmp/capstone/s26/run-t1-old-b2/SEQ). On PASS it writes CHAIN_NF_SQL_GO, which starts sql0-nf-treeB
# (tier3-sql-arm-rev2n.sh); on anything else it writes STOP, which holds it. Usage: chain-nf-b2.sh <sql0-frs out> <out>.
# Ends by itself. CZD_T1 stubs the boot for the dry run.
main() {
  set -u
  local W=${1:?sql0-frs out} O=${2:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local X=/tmp/capstone/s26 o n
  until grep -a -q "^SQL DONE\|^SQL STOP\|^SQL HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^SQL DONE\|^SQL STOP\|^SQL HOLD" "$W" 2>/dev/null || { echo "STOP: sql0-frs did not end" >> "$O"; return 1; }
  echo "CHAIN_NF BEGIN $(date +%T)" >> "$O"
  ${CZD_T1:-bash $S/../board-tier12/srz/run-tier1-nf-b2.sh} t1-nf > "$O.t1" 2>&1; echo "CHAIN_NF t1-nf END rc=$? $(date +%T)" >> "$O"
  o=$(grep -a '^RUNG ' $X/run-t1-old-b2/SEQ 2>/dev/null | sed 's/ oracle=.*//'); n=$(grep -a '^RUNG ' ${CZD_SEQ:-$X/run-t1-nf-b2/SEQ} 2>/dev/null | sed 's/ oracle=.*//')
  if [ -n "$n" ] && [ "$(echo "$n" | wc -l)" -eq 4 ] && [ "$o" = "$n" ]; then
    echo "CHAIN_NF T1 GATE PASS: four rungs read as t1-old's" >> "$O"; echo "CHAIN_NF_SQL_GO $(date +%T)" >> "$O"
  else echo "CHAIN_NF T1 GATE FAIL: old [$o] nf [$n]" >> "$O"; echo "STOP: the node-floor Tier 1 gate did not pass" >> "$O"; fi
  echo "CHAIN_NF DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
