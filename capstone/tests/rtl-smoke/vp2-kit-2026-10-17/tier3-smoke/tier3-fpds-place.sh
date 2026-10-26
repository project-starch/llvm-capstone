#!/bin/bash
# Placement of the rtl lane's fdiv/fsqrt rounding bare kit (2026-10-26; pre-registered in README.md): behind tonight's perf
# kits, on the FIRST board whose perf queue is done. Board 1's queue is k1, fpdiag, k3; board 2's is k2, k4; batch 2
# (b2k1) goes to whichever board ends its second kit first. A board is done when its second kit has ended AND batch 2 is
# placed on the other board or has ended. Batch 2 is placed the moment the first second kit ends, so before that neither
# board counts as done. Usage: tier3-fpds-place.sh <batch-1 log> <batch-2 log> <log>. FP_RUN stubs the boot.
main() {
  set -u
  local P1=${1:?batch-1 log} P2=${2:?batch-2 log} O=${3:?log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local t0=$(date +%s) b= p
  while :; do
    grep -q "^PM STOP\|^PM2 STOP" "$P1" "$P2" && { echo "FP STOP: the perf queue stopped" >> "$O"; return 1; }
    p=$(grep -a -o "^PM2 b2k1 on board [12]" "$P2" | head -1 | grep -o "[12]$")
    if [ -n "$p" ]; then
      local o=$((3-p)) k; [ $o = 1 ] && k=k3 || k=k4
      if grep -q "^PM $k on board $o END" "$P1"; then b=$o; break; fi
      if grep -q "^PM2 END" "$P2"; then b=$p; break; fi
    fi
    [ $(( $(date +%s)-t0 )) -gt 36000 ] && { echo "FP STOP: no board free in 10 h" >> "$O"; return 1; }; sleep 30
  done
  echo "FP fp-all on board $b (its perf queue is done) BEGIN $(date +%T)" >> "$O"
  local L=${FP_PASS:-/tmp/capstone/tree$([ $b = 1 ] && echo A || echo B)-session/pass}/fpds-b$b.out; : > $L
  if [ -n "${FP_RUN:-}" ]; then $FP_RUN $b $L; else bash $S/tier3-fpds-bare.sh $b $L 84488ed6db17; fi
  echo "FP END rc=$? $(date +%T) :: $(grep -a '^FPDS [A-Z]' $L | grep -v 'read:\|want:' | tail -1 | cut -c1-160)" >> "$O"
}
main "$@"; exit $?
