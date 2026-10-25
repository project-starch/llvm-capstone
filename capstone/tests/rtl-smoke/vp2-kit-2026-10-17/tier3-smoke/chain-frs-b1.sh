#!/bin/bash
# The fault-regs-srz module's own Tier 1 boot on BOARD 1 (tree A) (2026-10-20; pre-registered in README.md before the boot): after
# sql0-nf2A's out reads SQL DONE, STOP, HOLD, VOID or MUST-FIRE FAIL, run-tier1-frs-treeA-b1.sh t1-frs, then the gate: its four rung lines
# must equal t1-old's (/tmp/capstone/s26/run-t1-old-b2/SEQ). On PASS it writes CPF_GO, which starts cpyp-DF
# (tier3-cpy-f.sh); on anything else it writes STOP, which holds it. Usage: chain-frs-b1.sh <sql0-nf2A out> <out>.
# Ends by itself. CZD_T1 stubs the boot for the dry run.
main() {
  set -u
  local W=${1:?sql0-nf2A out} O=${2:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local X=/tmp/capstone/s26 o n
  until grep -a -q "^SQL DONE\|^SQL STOP\|^SQL HOLD\|^SQL VOID\|^SQL MUST-FIRE FAIL" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^SQL DONE\|^SQL STOP\|^SQL HOLD\|^SQL VOID\|^SQL MUST-FIRE FAIL" "$W" 2>/dev/null || { echo "STOP: sql0-nf2A did not end" >> "$O"; return 1; }
  echo "CHAIN_FRS BEGIN $(date +%T)" >> "$O"
  ${CZD_T1:-bash $S/../board-tier12/srz/run-tier1-frs-treeA-b1.sh} t1-frs > "$O.t1" 2>&1; echo "CHAIN_FRS t1-frs END rc=$? $(date +%T)" >> "$O"
  o=$(grep -a '^RUNG ' $X/run-t1-old-b2/SEQ 2>/dev/null | sed 's/ oracle=.*//'); n=$(grep -a '^RUNG ' ${CZD_SEQ:-$X/run-t1-frs-treeA-b1/SEQ} 2>/dev/null | sed 's/ oracle=.*//')
  if [ -n "$n" ] && [ "$(echo "$n" | wc -l)" -eq 4 ] && [ "$o" = "$n" ]; then
    echo "CHAIN_FRS T1 GATE PASS: four rungs read as t1-old's" >> "$O"; echo "CPF_GO $(date +%T)" >> "$O"
  else echo "CHAIN_FRS T1 GATE FAIL: old [$o] nf [$n]" >> "$O"; echo "STOP: the fault-regs-srz Tier 1 gate did not pass" >> "$O"; fi
  echo "CHAIN_FRS DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
