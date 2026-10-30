#!/bin/bash
# b4-boot.sh <board 1|2> <stage dir> <out dir> <step s> <booking min>: one boot of a stage on bitstream 4
# (caplifive_vp2c_325966ee4) through run-stage.py and boardvm. spec-board.sh with these changes:
# - the kit is make-b4-kit.sh's bkit13-b4-b<board>, checked by hash: module 6422805f49533ea5 (flat ids55), launcher
#   62753dd693cc4a25, and board.env naming caplifive_vp2c_325966ee4;
# - no --control: crc.K0 has no record on this launcher and module. The boot's known-good controls are the kit's own
#   board rungs, b0-stats and b0-hello (BOARD_RUNGS), which run before the stage and do not touch the virtual module;
# - the booking label names bitstream 4.
# SB_BUSY / SB_BOOK / SB_RUN stub the busy check, the hub and the runner; BOARDVM_STUB_TRANSCRIPT stubs the board.
main() {
  set -u
  local BRD=${1:?board} STAGE=${2:?stage} OUT=${3:?out} STEP=${4:?step s} BOOK=${5:?booking minutes}
  local H=/mnt/hdd/$USER/capstone K BUSY bid rc LOG RS
  RS=$H/wt-spec/capstone/ports/spec-cpu2017/board/run-stage.py
  K=${SB_KITROOT:-$H/t3-smoke}/bkit13-b4-b$BRD
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
  ( cd "$K" && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_325966ee4" "$K/board.env" \
    && [ "$(sha256sum "$K/adapter/module/capstone_vm.ko" | cut -c1-16)" = 6422805f49533ea5 ] \
    && [ "$(sha256sum "$K/adapter/capstone-vexec" | cut -c1-16)" = 62753dd693cc4a25 ] \
    || { echo "SB STOP: $K is not the bitstream-4 kit"; return 2; }
  [ -f "$STAGE/plan.json" ] && [ -x "$STAGE/stage/run.sh" ] || { echo "SB STOP: $STAGE has no plan.json or stage/run.sh"; return 2; }
  ${SB_BUSY:-$BUSY} > /dev/null || { echo "SB STOP: board $BRD is busy"; return 1; }
  bid=$(${SB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book "$BRD" "$BOOK" "vp2 bitstream 4 (B, ids55) $(basename "$STAGE") (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "SB STOP: the hub refused the booking (409)"; return 1; }
  [ $rc = 0 ] || { echo "SB NOTE: booking failed rc=$rc; continuing unbooked"; bid=; }
  echo "SB BEGIN $(date +%T) board $BRD kit ${K##*/} stage $(basename "$STAGE") booking ${bid:-none}"
  taskset -c 16-31 timeout $(( STEP * 4 + 3600 )) ${SB_RUN:-python3 $RS} run --backend board --kit "$K" \
    --stage "$STAGE" --out "$OUT" --step-s "$STEP" > "$LOG" 2>&1; rc=$?
  [ -n "$bid" ] && ${SB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook "$bid" > /dev/null 2>&1
  echo "SB END rc=$rc $(date +%T)"
  grep -E "^(RUN|READ)" "$LOG"
  return $rc
}
main "$@"; exit $?
