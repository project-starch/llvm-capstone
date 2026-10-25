#!/bin/bash
# q15 subset f on BOARD 2, tree B (2026-10-20; pre-registered in README.md before the boot): case 31's second bitstream. After
# cpyp-DB ends (its out reads CPP DONE, STOP or HOLD), writes the trigger tier3-q15-board2.sh waits for and runs it with
# Q15_SUBSETS=f and Q15_ROOT=t3-run-treeB/q15b (run dir q15bf). Usage: chain-q15bf-b2.sh <cpyp-DB out> <out>. Ends by itself.
# CQ_RUN stubs the run for the dry run.
main() {
  set -u
  local W=${1:?cpyp-DB out} O=${2:?out} S end=$(( $(date +%s) + 57600 )); S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  until grep -a -q "^CPP DONE\|^CPP STOP\|^CPP HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^CPP DONE\|^CPP STOP\|^CPP HOLD" "$W" 2>/dev/null || { echo "Q15 HOLD: cpyp-DB did not end" >> "$O"; return 1; }
  local G="$O.go"; echo "SQL DONE (bridge: cpyp-DB ended $(date +%T))" > "$G"
  Q15_SUBSETS=f Q15_ROOT=${CQ_ROOT:-/mnt/hdd/$USER/capstone/t3-run-treeB/q15b} ${CQ_RUN:-bash $S/tier3-q15-board2.sh} "$G" "$O"
}
main "$@"; exit $?
