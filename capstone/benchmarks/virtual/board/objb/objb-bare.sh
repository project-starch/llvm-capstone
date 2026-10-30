#!/bin/bash
# objb-bare.sh <board 1|2> <expected resident bitstream name, no .bit> <out dir>: the RTL lane's ten bare ObjBounds
# images (land/rtl-objb beb5384abb3b, capstone/tests/rtl-smoke/objb-bare-2026-10-30) on one board, in the pre-registered
# order (objb-pcc-ctrl, objb-stale, objb-perf-s1..s7, objb-pcc LAST), each through run_sup_bare.py (upload, power
# cycle, JTAG load at 0x80000000, continue, UART until SUPTEST END or SUP_TIMEOUT), then graded by value with
# objb-bare-grade.py. The arm is read from the name: caplifive_vp2c_e7af48457 is bitstream 3; anything else is graded
# as ObjBounds. Exit (land/rtl-objb 177c6fd9f6e9, Addendum 1): 0 AS EXPECTED (bitstream 3) or CONFIRMED (ObjBounds),
# 1 VOID, 3 STOP, 4 FINDING, 2 not run (a precondition failed) or no data. The grade words are in <out>/grade.txt.
# run_sup_bare.py is one image per boot (it power-cycles, loads and runs each), so objb-pcc, which wedges bitstream 3 in
# its trap handler, is alone in the LAST boot: SUP_TIMEOUT ends its wait, the runner then powers the board off and
# unlocks it, and `timeout 900` bounds the whole call.
# OBB_RUN / OBB_BUSY / OBB_BOOK stub the runner, the busy check and the hub for the dry run.
main() {
  set -u
  local BRD=${1:?board} BN=${2:?expected bitstream name} OUT=${3:?out dir}
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone} S ARM bid rc
  S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local K=${OBB_KIT:-$H/perf/objb/bare-kit}/images KSUMS=8dd2a4812c9da608
  local RUNNER=${OBB_RUNNER:-/tmp/capstone/wt-b2/capstone/tests/rtl-smoke/sup-bare-2026-10-03-ladder/run_sup_bare.py}
  local URLF BUSY L=$OUT/objb-bare.log
  case $BN in *.bit) echo "OBB STOP: give the name without .bit"; return 2;; esac
  case $BN in caplifive_vp2c_e7af48457) ARM=bs3;; caplifive_vp2c_*) ARM=objb;; *) echo "OBB STOP: $BN is not a caplifive_vp2c_ name"; return 2;; esac
  case $BRD in
    1) URLF=$HOME/.claude-kisp/secrets/fpga-console-url; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) URLF=$HOME/.claude-kisp/secrets/fpga-console-url-board2; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "OBB STOP: board $BRD"; return 2;;
  esac
  [ -e "$OUT" ] && { echo "OBB STOP: $OUT exists"; return 2; }
  mkdir -p "$OUT"
  echo "OBB START $(date '+%F %T') board $BRD bitstream $BN arm $ARM" >> "$L"
  # The images: the kit's SHA256SUMS must verify, and must be the RTL lane's at beb5384abb3b.
  ( cd "$K" && sha256sum -c --quiet SHA256SUMS ) && [ "$(sha256sum "$K/SHA256SUMS" | cut -c1-16)" = "${OBB_KSUMS:-$KSUMS}" ] \
    || { echo "OBB STOP: $K does not verify or is not the SHA256SUMS of beb5384abb3b (unchanged at 177c6fd9f6e9)" >> "$L"; return 2; }
  [ "$(sha256sum "$RUNNER" | cut -c1-16)" = "${OBB_RUNNER_SHA:-102ab84e762cba50}" ] || { echo "OBB STOP: $RUNNER is not run_sup_bare.py 102ab84e762cba50" >> "$L"; return 2; }
  ${OBB_BUSY:-$BUSY} > /dev/null || { echo "OBB STOP: board $BRD is busy" >> "$L"; return 2; }
  bid=$(${OBB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book "$BRD" 60 "vp2 objb-bare $ARM (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "OBB STOP: the hub refused the booking (409)" >> "$L"; return 2; }
  [ $rc = 0 ] || { echo "OBB NOTE: booking failed rc=$rc; continuing unbooked" >> "$L"; bid=; }
  for i in objb-pcc-ctrl objb-stale objb-perf-s1 objb-perf-s2 objb-perf-s3 objb-perf-s4 objb-perf-s5 objb-perf-s6 objb-perf-s7 objb-pcc; do
    echo "OBB IMAGE $i BEGIN $(date +%T) rec $(cat "$K/$i.rec")" >> "$L"
    ( export FPGA_URL; FPGA_URL=$(cat "$URLF")
      FPGA_BOARD=$BRD FPGA_BITSTREAM=$BN.bit SUP_IMG=$K/$i.bin SUP_REC_ADDR=$(cat "$K/$i.rec") SUP_OUT=$OUT/$i \
        SUP_TIMEOUT=${OBB_TIMEOUT:-90} timeout 900 ${OBB_RUN:-python3 $RUNNER} > "$OUT/$i.stderr" 2>&1 ); rc=$?
    echo "$rc" > "$OUT/$i.rc"
    echo "OBB IMAGE $i END rc=$rc $(date +%T)" >> "$L"
    grep -q 'HARD STOP' "$OUT/$i.stderr" "$OUT/$i/run.log" 2>/dev/null && { echo "OBB STOP: resident bitstream is not $BN.bit (HARD STOP at $i)" >> "$L"; break; }
  done
  [ -n "$bid" ] && ${OBB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook "$bid" > /dev/null 2>&1
  python3 "$S/objb-bare-grade.py" "$OUT" "$ARM" > "$OUT/grade.txt" 2>&1; rc=$?
  sed 's/^/  /' "$OUT/grade.txt" >> "$L"
  echo "OBB GRADE rc=$rc $(date +%T)" >> "$L"
  return $rc
}
main "$@"; exit $?
