#!/bin/bash
# Board 2 after sql0-frs-treeB (2026-10-20; pre-registered in README.md before the boot): the fetch-face sweep, ONE cold boot
# (board-tier12/ff/run-ff60-b2.sh). Waits for sql0-frs's out to read SQL DONE, STOP or HOLD, and for ff60-prep's log to read
# BP DONE (a BP STOP holds it). Usage: chain-ff60-b2.sh <sql0-frs out> <ff60-prep log> <out>. Ends by itself.
# CF_RUN stubs the boot for the dry run.
main() {
  set -u
  local W=${1:?sql0-frs out} P=${2:?prep log} O=${3:?out} S end=$(( $(date +%s) + 43200 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  until { grep -a -q "^SQL DONE\|^SQL STOP\|^SQL HOLD" "$W" && grep -a -q "^BP DONE\|^BP STOP" "$P"; } 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^SQL DONE\|^SQL STOP\|^SQL HOLD" "$W" 2>/dev/null || { echo "CHAIN_FF60 HOLD: sql0-frs did not end" >> "$O"; return 1; }
  grep -a -q "^BP DONE" "$P" 2>/dev/null || { echo "CHAIN_FF60 HOLD: ff60-prep did not finish: $(tail -1 "$P" 2>/dev/null)" >> "$O"; return 1; }
  echo "CHAIN_FF60 BEGIN $(date +%T)" >> "$O"
  ${CF_RUN:-bash $S/../board-tier12/ff/run-ff60-b2.sh} > "$O.run" 2>&1; echo "CHAIN_FF60 run END rc=$? $(date +%T)" >> "$O"
  echo "CHAIN_FF60 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
