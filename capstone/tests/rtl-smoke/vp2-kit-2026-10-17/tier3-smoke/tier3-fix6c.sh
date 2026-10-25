#!/bin/bash
# sq0's remaining subsets (a, f) on the S-27 workaround module, after the fix6b sequence (2026-10-18), so all six sq0
# subsets carry one module (d71f207a) and fold into one merged bundle; the fold maps an arm to ONE bundle. It reuses
# tier3-fix6-appvm.sh unchanged (kit bkit7-fix6, the subset invocation, scoring against sq0's prediction cut to each
# subset), starting it through a trigger file once the fix6b sequence ends.
# Usage: tier3-fix6c.sh <fix6b out> <out>. Ends by itself. FIX6C_TRIGGER replaces the trigger file for the dry run.
main() {
  set -u
  local W=${1:?fix6b out} O=${2:?out}; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local T=${FIX6C_TRIGGER:-/mnt/hdd/$USER/capstone/t3-run-fix6c.trigger} end=$(( $(date +%s) + 43200 ))
  until grep -a -q "^FIX6BSEQ DONE\|^FIX6BSEQ HOLD\|^FIX6BSEQ STOP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  grep -a -q "^FIX6BSEQ" "$W" || { echo "FIX6C HOLD: the fix6b sequence did not end" >> "$O"; return 1; }
  echo "FIX6SEQ DONE (trigger for sq0a/sq0f, after: $(grep -a '^FIX6BSEQ' "$W" | tail -1))" > "$T"
  bash "$S/tier3-fix6-appvm.sh" "$T" "$O" sq0a sq0f
}
main "$@"; exit $?
