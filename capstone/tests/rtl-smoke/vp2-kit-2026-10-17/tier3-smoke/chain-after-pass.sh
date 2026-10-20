#!/bin/bash
# chain-after-pass.sh <previous out> <tag> <out> -- <tier3-pass-appvm.sh args...>: wait until the previous pass arm's out
# reads "PASS <tag> DONE" (start) or a PASS STOP (hold), bounded at 12 h; then run tier3-pass-appvm.sh with the given
# arguments. A HOLD writes one line and starts nothing. Ends by itself.
main() {
  set -u
  local W=${1:?previous out} TAG=${2:?previous tag} O=${3:?out} end=$(( $(date +%s) + 43200 )); shift 3; [ "${1:-}" = -- ] && shift
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  until grep -a -q "^PASS $TAG DONE\|^PASS STOP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  grep -a -q "^PASS $TAG DONE" "$W" || { echo "PASS HOLD: $TAG did not end DONE ($(tail -1 "$W" 2>/dev/null))" >> "$O"; return 1; }
  echo "CHAIN after $TAG DONE $(date +%T)" >> "$O"
  bash "$S/tier3-pass-appvm.sh" "$@"
}
main "$@"; exit $?
