#!/bin/bash
# spec-board.sh <board 1|2> <stage dir> <out dir> <step s> <booking min>: one boot of a make-objb-stages.py stage on the
# ObjBounds bitstream, through run-stage.py and boardvm. A copy of the SPEC lane's spec-board.sh
# (capstone/ports/spec-cpu2017/board, 0690c056b40883ff) with three changes, so the live bitstream-3 chain's own copy is
# never edited:
# - the kit is SB_KIT (required), not bkit12-perf-b3-b<board>;
# - its board.env must name SB_BITNAME (required) exactly, and SB_BITNAME must not be bitstream 3;
# - the runner is the SPEC lane's run-stage.py by path (SB_RUNSTAGE), checked by hash (SB_RUNSTAGE_SHA, 16 hex).
# The module and launcher must still be the perf kit's (9c60fb53f80ce67e, 978c09d61f774315), and the boot runs the same
# known-good control first: crc.K0.dom of the perf main run, T1, sha256 bd909cc6eba4bf2d... (W 1, I 5).
# SB_BUSY / SB_BOOK / SB_RUN stub the busy check, the hub and the runner for the dry run; BOARDVM_STUB_TRANSCRIPT,
# read by boardvm, stubs the board itself.
main() {
  set -u
  local BRD=${1:?board} STAGE=${2:?stage} OUT=${3:?out} STEP=${4:?step s} BOOK=${5:?booking minutes}
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone} BUSY bid rc LOG
  local K=${SB_KIT:?SB_KIT: the ObjBounds board kit} BN=${SB_BITNAME:?SB_BITNAME: the resident bitstream name}
  local RS=${SB_RUNSTAGE:-$H/wt-spec/capstone/ports/spec-cpu2017/board/run-stage.py} RSH=${SB_RUNSTAGE_SHA:-734ad25cf901d4da}
  local CTL=$H/perf/build2/T1/crc.K0.dom
  LOG=$OUT.log
  case $BRD in
    1) BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "SB STOP: board $BRD"; return 2;;
  esac
  [ -e "$OUT" ] && { echo "SB STOP: $OUT exists"; return 2; }
  if [ -n "${SB_BUSY:-}${SB_BOOK:-}${SB_RUN:-}" ] && [ -z "${BOARDVM_STUB_TRANSCRIPT:-}" ] && [ -z "${SB_RUN:-}" ]; then
    echo "SB STOP: a stubbed dry run without BOARDVM_STUB_TRANSCRIPT or SB_RUN would reach the real board"; return 2
  fi
  case $BN in *e7af48457*) echo "SB STOP: SB_BITNAME $BN is bitstream 3, not the ObjBounds variant"; return 2;; esac
  ( cd "$K" && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$BN" "$K/board.env" \
    && [ "$(sha256sum "$K/adapter/module/capstone_vm.ko" | cut -c1-16)" = 9c60fb53f80ce67e ] \
    && [ "$(sha256sum "$K/adapter/capstone-vexec" | cut -c1-16)" = 978c09d61f774315 ] \
    || { echo "SB STOP: $K is not the ObjBounds perf kit for $BN"; return 2; }
  [ "$(sha256sum "$RS" | cut -c1-16)" = "$RSH" ] || { echo "SB STOP: $RS is not run-stage.py $RSH"; return 2; }
  [ "$(sha256sum "$CTL" | cut -c1-16)" = bd909cc6eba4bf2d ] || { echo "SB STOP: the control image is not crc.K0 T1 bd909cc6"; return 2; }
  [ -f "$STAGE/plan.json" ] && [ -x "$STAGE/stage/run.sh" ] || { echo "SB STOP: $STAGE is not an armed make-objb-stages.py output"; return 2; }
  ${SB_BUSY:-$BUSY} > /dev/null || { echo "SB STOP: board $BRD is busy"; return 1; }
  bid=$(${SB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book "$BRD" "$BOOK" "vp2 ObjBounds $(basename "$STAGE") (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "SB STOP: the hub refused the booking (409)"; return 1; }
  [ $rc = 0 ] || { echo "SB NOTE: booking failed rc=$rc; continuing unbooked"; bid=; }
  echo "SB BEGIN $(date +%T) board $BRD kit ${K##*/} stage $(basename "$STAGE") booking ${bid:-none}"
  taskset -c 16-31 timeout $(( STEP * 4 + 3600 )) ${SB_RUN:-python3 $RS} run --backend board --kit "$K" \
    --stage "$STAGE" --out "$OUT" --tree "${SB_TREE:-$H/spec/tree-board/SPEC17}" --control "$CTL" 1 5 --step-s "$STEP" \
    ${SB_TWIN:+--twin "$SB_TWIN"} > "$LOG" 2>&1; rc=$?
  [ -n "$bid" ] && ${SB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook "$bid" > /dev/null 2>&1
  echo "SB END rc=$rc $(date +%T)"
  grep -E "^(RUN|READ)" "$LOG"
  return $rc
}
main "$@"; exit $?
