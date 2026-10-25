#!/bin/bash
# chain-after.sh <previous out> <tag> <out> -- <script> <args...>: wait until the previous pass arm's out reads
# "PASS <tag> DONE" (start) or a PASS STOP / PASS HOLD (hold), bounded at 12 h; then run <script> (a path under this
# directory) with <args...>. A hold writes one line and starts nothing. chain-after-pass.sh with the script as an argument.
main() {
  set -u
  local W=${1:?previous out} TAG=${2:?previous tag} O=${3:?out} end=$(( $(date +%s) + 43200 )); shift 3; [ "${1:-}" = -- ] && shift
  local S X=${1:?script}; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); shift
  until grep -a -q "^PASS $TAG DONE\|^PASS STOP\|^PASS HOLD" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  grep -a -q "^PASS $TAG DONE" "$W" || { echo "PASS HOLD: $TAG did not end DONE ($(tail -1 "$W" 2>/dev/null))" >> "$O"; return 1; }
  echo "CHAIN after $TAG DONE $(date +%T)" >> "$O"
  bash "$S/$X" "$@"
}
main "$@"; exit $?
