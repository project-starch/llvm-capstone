#!/bin/bash
# spec-board.sh <board 1|2> <stage dir> <out dir> <step s> <booking min>: one boot of a make-stage.py stage on the vp2
# board, bitstream 3, through run-stage.py and boardvm.
# The board kit is the perf evaluation's bkit12-perf-b3-b<board> (launcher 978c09d61f774315, module 9c60fb53f80ce67e
# with the bench counters), checked here by hash. The boot runs the known-good control first: crc.K0.dom of the perf
# main run, T1, sha256 bd909cc6eba4bf2d..., which passed on bitstream 3 in kit t3a2 (W 1, I 5); then the SPEC step.
# SB_BUSY / SB_BOOK / SB_RUN stub the busy check, the hub and the runner for the dry run (the board-run skill's
# rule 8); BOARDVM_STUB_TRANSCRIPT, read by boardvm, stubs the board itself.
main() {
  set -u
  local BRD=${1:?board} STAGE=${2:?stage} OUT=${3:?out} STEP=${4:?step s} BOOK=${5:?booking minutes}
  local H=/mnt/hdd/$USER/capstone HERE K BUSY bid rc LOG
  HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  K=${SB_KITROOT:-$H/t3-smoke}/bkit12-perf-b3-b$BRD
  local CTL=$H/perf/build2/T1/crc.K0.dom
  LOG=$OUT.log
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "SB STOP: board $BRD"; return 2;;
  esac
  [ -e "$OUT" ] && { echo "SB STOP: $OUT exists"; return 2; }
  # A dry run stubs EVERY board call or none: a stubbed busy check or hub with a live board recipe reaches the real
  # driver on any branch that continues (2026-10-28: a broken booking stub fell through to "continuing unbooked" and
  # the real recipe ran; the board lock, held by another lane, refused it).
  if [ -n "${SB_BUSY:-}${SB_BOOK:-}${SB_RUN:-}" ] && [ -z "${BOARDVM_STUB_TRANSCRIPT:-}" ] && [ -z "${SB_RUN:-}" ]; then
    echo "SB STOP: a stubbed dry run without BOARDVM_STUB_TRANSCRIPT or SB_RUN would reach the real board"; return 2
  fi
  ( cd "$K" && sha256sum -c --quiet SHA256SUMS ) && grep -q "^BOARD_BITSTREAM=caplifive_.*e7af48457$" "$K/board.env" \
    && [ "$(sha256sum "$K/adapter/module/capstone_vm.ko" | cut -c1-16)" = 9c60fb53f80ce67e ] \
    && [ "$(sha256sum "$K/adapter/capstone-vexec" | cut -c1-16)" = 978c09d61f774315 ] \
    || { echo "SB STOP: $K is not the bitstream-3 perf kit"; return 2; }
  [ "$(sha256sum "$CTL" | cut -c1-16)" = bd909cc6eba4bf2d ] || { echo "SB STOP: the control image is not crc.K0 T1 bd909cc6"; return 2; }
  [ -f "$STAGE/plan.json" ] && [ -x "$STAGE/stage/run.sh" ] || { echo "SB STOP: $STAGE is not a make-stage.py output"; return 2; }
  ${SB_BUSY:-$BUSY} > /dev/null || { echo "SB STOP: board $BRD is busy"; return 1; }
  bid=$(${SB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book "$BRD" "$BOOK" "vp2 bitstream 3 SPEC $(basename "$STAGE") (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "SB STOP: the hub refused the booking (409)"; return 1; }
  [ $rc = 0 ] || { echo "SB NOTE: booking failed rc=$rc; continuing unbooked"; bid=; }
  echo "SB BEGIN $(date +%T) board $BRD kit ${K##*/} stage $(basename "$STAGE") booking ${bid:-none}"
  taskset -c 16-31 timeout $(( STEP * 4 + 3600 )) ${SB_RUN:-python3 $HERE/run-stage.py} run --backend board --kit "$K" \
    --stage "$STAGE" --out "$OUT" --tree "${SB_TREE:-$H/spec/tree-board/SPEC17}" --control "$CTL" 1 5 --step-s "$STEP" \
    ${SB_TWIN:+--twin "$SB_TWIN"} > "$LOG" 2>&1; rc=$?
  [ -n "$bid" ] && ${SB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook "$bid" > /dev/null 2>&1
  echo "SB END rc=$rc $(date +%T)"
  grep -E "^(RUN|READ)" "$LOG"
  return $rc
}
main "$@"; exit $?
