#!/bin/bash
# r32-c5a.sh <board 1|2> <out dir>: the rtl lane's R-32 SHRINKTO readback (llvm-capstone land/rtl-r32-kit 624924575c53,
# capstone/tests/rtl-smoke/r32-shrinkto-bare-2026-11-06, image sha256 34311f6d...) as one bare boot on C.5a:
# flash C.5a (sealed sha256, read back), book the board, run_sup_bare.py with the image and its recorder address,
# unbook, grade with r32-grade.py --expect prefix. The board is left on C.5a. Exit = the grader's (0 AS EXPECTED,
# 1 the other vector or a finding, 2 VOID / no data), or 3 when a precondition or the flash fails.
# Stubs: R32_BOOK R32_FLASH R32_READBACK R32_RUN.
main() {
  set -u
  local BRD=${1:?board} OUT=${2:?out dir}
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}; local K=$H/perf/r32/kit-62492457 L=$OUT/r32.log
  local C5=caplifive_vp2c_1fc2cd5aa C5SHA=5b7b1edcd9bcd2521bd4bd59107747e268487056f1bfcd96403fd612eeb6d764
  local T=$H/wt-spec/capstone/tests/rtl-smoke SB=$H/wt-spec/capstone/ports/spec-cpu2017
  local RUNNER=/tmp/capstone/wt-b2/capstone/tests/rtl-smoke/sup-bare-2026-10-03-ladder/run_sup_bare.py
  local BOOK=${R32_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} FLASH=${R32_FLASH:-python3 $T/drivers/flash-bitstream.py}
  local RB=${R32_READBACK:-python3 $SB/results/2026-10-28-board-test/b4/nv-readback.py} URLF b rc t0
  case $BRD in 1) URLF=$HOME/.claude-kisp/secrets/fpga-console-url;; 2) URLF=$HOME/.claude-kisp/secrets/fpga-console-url-board2;; *) return 3;; esac
  [ -e "$OUT" ] && { echo "R32 STOP: $OUT exists"; return 3; }
  mkdir -p "$OUT"; log() { echo "R32 $(date '+%F %T') $*" >> "$L"; }; log "START pid $$ board $BRD"
  [ -s "$URLF" ] || { log "STOP: no console URL file"; return 3; }
  ( cd "$K/images" && sha256sum -c --quiet SHA256SUMS ) && grep -q '^34311f6d26259cb19b6ef06588e8d22f2a20534287d63f6a7348d3bc97c26863 ' "$K/images/SHA256SUMS" \
    || { log "STOP: kit image does not verify"; return 3; }
  [ "$(sha256sum "$RUNNER" | cut -c1-16)" = 102ab84e762cba50 ] || { log "STOP: runner hash"; return 3; }
  [ "$(sha256sum "$H/bitstreams/$C5.bit" | cut -d' ' -f1)" = $C5SHA ] || { log "STOP: C.5a .bit hash"; return 3; }
  export FPGA_BOARD=$BRD CAPSTONE_FPGA_URL_FILE=$URLF BOARD_URL_FILE=$URLF
  b=$($BOOK book $BRD 30 "vp2 $C5 REFLASH (board $BRD, R-32 kit)") || b=
  log "FLASH BEGIN booking ${b:-none}"; $FLASH "$C5.bit" "$H/bitstreams/$C5.bit" $C5SHA >> "$L" 2>&1; rc=$?
  [ $rc = 0 ] || { log "FLASH rc=$rc, retrying once in 60 s"; sleep "${FLASH_RETRY_S:-60}"; $FLASH "$C5.bit" "$H/bitstreams/$C5.bit" $C5SHA >> "$L" 2>&1; rc=$?; }
  [ -n "$b" ] && $BOOK unbook "$b" >> "$L" 2>&1; log "FLASH END rc=$rc"; [ $rc = 0 ] || return 3
  $RB "$C5.bit" $C5SHA >> "$L" 2>&1; rc=$?; case $rc in 0|3) log "READBACK ok rc=$rc";; *) log "READBACK rc=$rc"; return 3;; esac
  t0=$(date +%s)
  while :; do b=$($BOOK book $BRD 20 "vp2 R-32 SHRINKTO bare kit (board $BRD)"); rc=$?; [ $rc = 0 ] && break
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { log "STOP: no booking in 1800 s"; return 3; }; sleep 30; done
  log "BOOKED $b; RUN BEGIN"
  ( export FPGA_URL; FPGA_URL=$(cat "$URLF")
    FPGA_BITSTREAM=$C5.bit SUP_IMG=$K/images/shrinkto.bin SUP_REC_ADDR=$(cat "$K/images/shrinkto.rec") SUP_OUT=$OUT/run \
      SUP_TIMEOUT=90 timeout 900 ${R32_RUN:-python3 $RUNNER} > "$OUT/run.stderr" 2>&1 ); rc=$?
  $BOOK unbook "$b" >> "$L" 2>&1; log "RUN END rc=$rc; unbooked"
  grep -q 'HARD STOP' "$OUT/run.stderr" "$OUT/run/run.log" 2>/dev/null && log "HARD STOP in the runner (resident is not $C5?)"
  if [ -s "$OUT/run/uart.txt" ]; then python3 "$K/r32-grade.py" "$OUT/run/uart.txt" --expect prefix > "$OUT/grade.txt" 2>&1; rc=$?
  else echo "NO DATA: $OUT/run/uart.txt is missing or empty" > "$OUT/grade.txt"; rc=2; fi
  sed 's/^/  /' "$OUT/grade.txt" >> "$L"; log "GRADE rc=$rc"; log "DONE r32 rc=$rc; board $BRD left on $C5"
  return $rc
}
main "$@"; exit $?
