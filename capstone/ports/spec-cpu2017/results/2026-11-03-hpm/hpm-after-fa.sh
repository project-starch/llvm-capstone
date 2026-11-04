#!/bin/bash
# hpm-after-fa.sh <fa-b1.sh pid>: after board 1's fault-ahead run ends, the two counter boots of hpm PREREG Addendum 2
# (imagick dev and devmk eval stages, kit bkit16-hpm-b3-b1, sbhpm/chain-board.sh) into spec/board/out-hpm-eval.
main() {
  set -u
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone} PID=${1:?fa-b1.sh pid} L t=0
  L=$H/perf/fa/hpm-after-fa.log
  echo "HA START $(date '+%F %T') waiting for pid $PID" >> "$L"
  while kill -0 "$PID" 2>/dev/null; do [ $t -ge 172800 ] && { echo "HA STOP: pid $PID still running" >> "$L"; return 2; }; sleep 60; t=$((t+60)); done
  grep -q "^FA .* DONE" "$H/perf/fa/fa-b1.log" || { echo "HA STOP: fa-b1.sh ended without DONE" >> "$L"; return 2; }
  echo "HA BEGIN $(date +%T)" >> "$L"
  ${HA_CHAIN:-bash $H/perf/hpm/sbhpm/chain-board.sh} 1 "$H/perf/fa/hpm-eval-b1.list" "$H/spec/board/out-hpm-eval" >> "$L" 2>&1
  echo "HA END rc=$? $(date +%T)" >> "$L"
}
main "$@"; exit $?
