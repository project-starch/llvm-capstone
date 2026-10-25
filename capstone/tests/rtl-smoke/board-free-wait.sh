#!/bin/bash
# board-free-wait.sh <bound_s> [board]: wait until board 1 or 2 (default 1) is free, by board-busy.py (its host lock not
# held, and no board-user process naming it). Exit 0 free, 1 at the bound, 2 on a bad board id. Per board since
# 2026-10-19 (board #2, phase 1): a run on one board never holds a chain waiting for the other.
main() {
  set -u
  local bound=${1:?bound_s} b=${2:-1} end G; G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/board-busy.py
  case $b in 1|2) ;; *) echo "board-free-wait: board must be 1 or 2, got $b"; return 2;; esac
  end=$(( $(date +%s) + bound ))
  until python3 "$G" "$b" > /dev/null; do
    [ "$(date +%s)" -ge "$end" ] && return 1
    sleep "${BOARD_POLL_S:-30}"
  done
  return 0
}
main "$@"; exit $?
