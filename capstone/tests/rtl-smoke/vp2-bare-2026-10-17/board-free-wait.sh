#!/bin/bash
# board-free-wait.sh <bound_s>: wait until no board user is running, ignoring this script's OWN ancestors. Exit 0 when
# the board is free, 1 at the bound. A board user is a bare or baked runner (or the infra-retry wrapper around one), or
# a session script under vp1-session/, st100-1016/ or vp1-flash/.
# Why the ancestors: chain-vp2.sh's inline wait matched `bash /tmp/capstone/vp1-session/rerun-then-vp2.sh`, which is
# the chain's own parent when it runs from there, and that wait had no bound (found 2026-10-17 before it could fire).
main() {
  set -u
  local bound=${1:?bound_s}
  local pat=${BOARD_BUSY_RE:-'^(timeout [0-9]+ )?(\S*/)?python3 (-m fpga_driver|\S*run_sup_bare(_wedge)?\.py|\S*infra-retry\.py)|^bash /tmp/capstone/(vp1-session|st100-1016|vp1-flash)/'}
  local anc=" $$ " p=$$ end=$(( $(date +%s) + bound ))
  while p=$(ps -o ppid= -p "$p" | tr -d ' ') && [ -n "$p" ] && [ "$p" -gt 1 ]; do anc="$anc$p "; done
  while ps -eo pid=,args= | awk -v a="$anc" '{pid=$1; sub(/^ *[0-9]+ /, ""); if (index(a, " " pid " ") == 0) print}' \
        | grep -q -E "$pat"; do
    [ "$(date +%s)" -ge "$end" ] && return 1
    sleep "${BOARD_POLL_S:-30}"
  done
  return 0
}
main "$@"; exit $?
