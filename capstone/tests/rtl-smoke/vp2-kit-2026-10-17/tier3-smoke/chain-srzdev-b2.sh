#!/bin/bash
# dev's srevroot-zero module's own Tier 1 boot on BOARD 2 (2026-10-20; pre-registered in README.md before the boot): after the
# fetch-face boot ends (its SEQ reads FF_DONE_RUN, STOP or NO DATA), run-tier1-srzdev-b2.sh t1-srzdev, then the gate: its four
# rung lines must equal t1-old's (/tmp/capstone/s26/run-t1-old-b2/SEQ). Usage: chain-srzdev-b2.sh <out>. Ends by itself.
main() {
  set -u
  local O=${1:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local F=/tmp/capstone/s26/run-ff-b2/SEQ X=/tmp/capstone/s26 o n
  until grep -a -q "^FF_DONE_RUN\|^STOP\|^NO DATA" $F 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  echo "CHAIN_SRZDEV BEGIN $(date +%T)" >> "$O"
  ${CZD_T1:-bash $S/../board-tier12/srz/run-tier1-srzdev-b2.sh} t1-srzdev > "$O.t1" 2>&1; echo "CHAIN_SRZDEV t1-srzdev END rc=$? $(date +%T)" >> "$O"
  o=$(grep -a '^RUNG ' $X/run-t1-old-b2/SEQ 2>/dev/null | sed 's/ oracle=.*//'); n=$(grep -a '^RUNG ' $X/run-t1-srzdev-b2/SEQ 2>/dev/null | sed 's/ oracle=.*//')
  if [ -n "$n" ] && [ "$(echo "$n" | wc -l)" -eq 4 ] && [ "$o" = "$n" ]; then echo "CHAIN_SRZDEV T1 GATE PASS: four rungs read as t1-old's" >> "$O"
  else echo "CHAIN_SRZDEV T1 GATE FAIL: old [$o] srzdev [$n]" >> "$O"; fi
  echo "CHAIN_SRZDEV DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
