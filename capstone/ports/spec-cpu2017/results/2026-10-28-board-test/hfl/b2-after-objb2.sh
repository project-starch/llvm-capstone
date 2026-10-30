#!/bin/bash
# b2-after-objb.sh <objb-first pid>: board 2 after the ObjBounds kits. Waits for objb-first.sh to exit; goes on only if
# its log carries "OB MAIN END rc=0" (any other end leaves board 2 on ObjBounds for a look); runs the RTL lane's
# one-image witness if its hand-off is ready (objb-witness.sh; any outcome, including SKIP, goes on); reflashes bitstream 3 with
# b2-to-bs3.sh (rc 0 = RB DONE); waits until PREREG Addendum 18 is on origin/spec-silicon (never boots unregistered);
# then runs train-hfl-b2.lines through chain-board.sh on the standard kit (bkit12-perf-b3-b2). Log: out/b2-after-objb.log.
# Stubs: AO_OUT, AO_OBLOG, AO_OW, AO_RB, AO_PREREG, AO_CB (commands), AO_POLL, AO_WAITMAX, AO_PREMAX.
main() {
  set -u
  local OPID=${1:?objb-first pid} W=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone} HERE r t0
  HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local O=${AO_OUT:-$W/spec/board/out} LIST=$W/spec/board/train-hfl-b2.lines OBLOG=${AO_OBLOG:-$W/perf/objb/objb-first.log}
  local K=$W/wt-spec/capstone/ports/spec-cpu2017 P=capstone/ports/spec-cpu2017/results/2026-10-28-board-test/PREREG.md
  local L=$O/b2-after-objb.log
  local OW=${AO_OW:-bash $HERE/objb-witness.sh $W/perf/objb/board/out/witness}
  local RB=${AO_RB:-bash $HERE/b2-to-bs3.sh $L}
  local PRE=${AO_PREREG:-prereg18}
  local CB=${AO_CB:-bash $K/board/chain-board.sh}
  prereg18() { git -C "$W/wt-spec" fetch -q origin spec-silicon && git -C "$W/wt-spec" show "origin/spec-silicon:$P" | grep -q '^## Addendum 18'; }
  echo "$(date '+%F %T') AO waiting for objb-first pid $OPID" >> "$L"
  t0=$(date +%s)
  while kill -0 "$OPID" 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${AO_WAITMAX:-86400} ] && { echo "$(date '+%F %T') AO STOP: objb-first still running" >> "$L"; return 1; }
    sleep ${AO_POLL:-60}
  done
  grep -q '^OB MAIN END rc=0' "$OBLOG" || { echo "$(date '+%F %T') AO STOP: no 'OB MAIN END rc=0' in $OBLOG; board 2 left on ObjBounds" >> "$L"; return 1; }
  echo "$(date '+%F %T') AO ObjBounds main ended rc=0; witness" >> "$L"
  $OW; r=$?
  echo "$(date '+%F %T') AO witness rc=$r (0 ran, 2 not run, 1 runner failed); reflashing bitstream 3" >> "$L"
  $RB; r=$?
  [ $r = 0 ] || { echo "$(date '+%F %T') AO STOP: reflash rc=$r" >> "$L"; return 1; }
  t0=$(date +%s)
  until $PRE; do
    [ $(( $(date +%s) - t0 )) -gt ${AO_PREMAX:-43200} ] && { echo "$(date '+%F %T') AO STOP: Addendum 18 never reached origin/spec-silicon" >> "$L"; return 1; }
    echo "$(date '+%F %T') AO waiting for Addendum 18 on origin/spec-silicon" >> "$L"
    sleep ${AO_PREPOLL:-600}
  done
  echo "$(date '+%F %T') AO Addendum 18 registered; chain b2 on $LIST" >> "$L"
  rm -f "$O/.chain-b2-done" "$O/.chain-b2-stopped"
  $CB 2 "$LIST" "$O" >> "$L" 2>&1; r=$?
  echo "$(date '+%F %T') AO chain b2 rc=$r" >> "$L"
  return $r
}
main "$@"; exit $?
