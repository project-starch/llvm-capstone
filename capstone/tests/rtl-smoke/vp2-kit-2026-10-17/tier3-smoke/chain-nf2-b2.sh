#!/bin/bash
# The node-floor-2 module's own Tier 1 boot on BOARD 2 (2026-10-20; pre-registered in README.md before the boot): after
# sql0-srzdev-treeB ends (its out reads SQL DONE, STOP or HOLD), run-tier1-nf2-b2.sh t1-nf2, then the gate: its four rung lines
# must equal t1-old's (/tmp/capstone/s26/run-t1-old-b2/SEQ). On PASS it writes CHAIN_NF2_SQL_GO, which starts sql0-nf2-treeB
# (tier3-sql-arm-rev2p.sh); on anything else it writes STOP, which holds it. Usage: chain-nf2-b2.sh <sql0-srzdev out> <out>.
# Ends by itself. CZD_T1 stubs the boot for the dry run.
main() {
  set -u
  local W=${1:?sql0-srzdev out} O=${2:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local X=/tmp/capstone/s26 o n
  until grep -a -q "^SQL DONE\|^SQL STOP\|^SQL HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^SQL DONE\|^SQL STOP\|^SQL HOLD" "$W" 2>/dev/null || { echo "STOP: sql0-srzdev did not end" >> "$O"; return 1; }
  echo "CHAIN_NF2 BEGIN $(date +%T)" >> "$O"
  ${CZD_T1:-bash $S/../board-tier12/srz/run-tier1-nf2-b2.sh} t1-nf2 > "$O.t1" 2>&1; echo "CHAIN_NF2 t1-nf2 END rc=$? $(date +%T)" >> "$O"
  o=$(grep -a '^RUNG ' $X/run-t1-old-b2/SEQ 2>/dev/null | sed 's/ oracle=.*//'); n=$(grep -a '^RUNG ' ${CZD_SEQ:-$X/run-t1-nf2-b2/SEQ} 2>/dev/null | sed 's/ oracle=.*//')
  if [ -n "$n" ] && [ "$(echo "$n" | wc -l)" -eq 4 ] && [ "$o" = "$n" ]; then
    echo "CHAIN_NF2 T1 GATE PASS: four rungs read as t1-old's" >> "$O"; echo "CHAIN_NF2_SQL_GO $(date +%T)" >> "$O"
  else echo "CHAIN_NF2 T1 GATE FAIL: old [$o] nf [$n]" >> "$O"; echo "STOP: the node-floor Tier 1 gate did not pass" >> "$O"; fi
  echo "CHAIN_NF2 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
