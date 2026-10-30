#!/bin/bash
# objb-witness.sh <out dir>: the RTL lane's one-image ObjBounds witness (objb-stale-w) on board 2, run on the ObjBounds
# bitstream after its main kits and before the reflash back to bitstream 3. Same runner, env and bounds as objb-bare.sh
# (run_sup_bare.py 102ab84e762cba50: upload, power cycle, JTAG load, UART until SUPTEST END or SUP_TIMEOUT; timeout 900).
# Hand-off contract, in $CAPSTONE_WORK/perf/objb/witness/: objb-stale-w.bin, objb-stale-w.rec, SHA256SUMS over both,
# READY = one line, the commit carrying its pre-registration. Runs only if SHA256SUMS verifies and that commit is on
# origin/land/rtl-objb (pre-registered before the boot). Exit: 0 ran (the RTL lane grades), 2 not run (precondition),
# 1 runner rc != 0. Stubs: OW_DIR, OW_RUN, OW_BUSY, OW_BOOK, OW_PRE.
main() {
  set -u
  local OUT=${1:?out dir} H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone} BRD=2 BN=caplifive_vp2c_603528b8b bid rc c
  local D=${OW_DIR:-$H/perf/objb/witness} RUNNER=/tmp/capstone/wt-b2/capstone/tests/rtl-smoke/sup-bare-2026-10-03-ladder/run_sup_bare.py
  local URLF=$HOME/.claude-kisp/secrets/fpga-console-url-board2 L=$OUT/witness.log
  pre() { git -C /home/$USER/dev/llvm-capstone fetch -q origin land/rtl-objb && git -C /home/$USER/dev/llvm-capstone merge-base --is-ancestor "$1" FETCH_HEAD; }
  [ -e "$OUT" ] && { echo "OW STOP: $OUT exists"; return 2; }
  mkdir -p "$OUT"
  echo "OW START $(date '+%F %T') board $BRD bitstream $BN" >> "$L"
  [ -f "$D/READY" ] || { echo "OW SKIP: no $D/READY" >> "$L"; return 2; }
  ( cd "$D" && sha256sum -c --quiet SHA256SUMS && grep -q ' objb-stale-w.bin$' SHA256SUMS && grep -q ' objb-stale-w.rec$' SHA256SUMS ) \
    || { echo "OW STOP: $D/SHA256SUMS does not verify or does not cover the image and its .rec" >> "$L"; return 2; }
  c=$(head -1 "$D/READY" | tr -d ' \n')
  [[ $c =~ ^[0-9a-f]{12,40}$ ]] || { echo "OW STOP: READY is not a commit hash" >> "$L"; return 2; }
  ${OW_PRE:-pre} "$c" || { echo "OW STOP: $c is not on origin/land/rtl-objb (not pre-registered)" >> "$L"; return 2; }
  [ "$(sha256sum "$RUNNER" | cut -c1-16)" = 102ab84e762cba50 ] || { echo "OW STOP: runner is not 102ab84e762cba50" >> "$L"; return 2; }
  ${OW_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2} > /dev/null || { echo "OW STOP: board 2 busy" >> "$L"; return 2; }
  bid=$(${OW_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 20 "vp2 objb witness (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "OW STOP: the hub refused the booking (409)" >> "$L"; return 2; }
  [ $rc = 0 ] || { echo "OW NOTE: booking failed rc=$rc; continuing unbooked" >> "$L"; bid=; }
  cp -p "$D/SHA256SUMS" "$D/READY" "$OUT/"
  echo "OW IMAGE objb-stale-w BEGIN $(date +%T) prereg $c rec $(cat "$D/objb-stale-w.rec")" >> "$L"
  ( export FPGA_URL; FPGA_URL=$(cat "$URLF")
    FPGA_BOARD=$BRD FPGA_BITSTREAM=$BN.bit SUP_IMG=$D/objb-stale-w.bin SUP_REC_ADDR=$(cat "$D/objb-stale-w.rec") \
      SUP_OUT=$OUT/objb-stale-w SUP_TIMEOUT=90 timeout 900 ${OW_RUN:-python3 $RUNNER} > "$OUT/objb-stale-w.stderr" 2>&1 ); rc=$?
  echo "$rc" > "$OUT/objb-stale-w.rc"
  echo "OW IMAGE objb-stale-w END rc=$rc $(date +%T)" >> "$L"
  [ -n "$bid" ] && ${OW_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook "$bid" > /dev/null 2>&1
  grep -q 'HARD STOP' "$OUT/objb-stale-w.stderr" "$OUT/objb-stale-w/run.log" 2>/dev/null && echo "OW NOTE: HARD STOP (resident is not $BN.bit)" >> "$L"
  [ $rc = 0 ] || return 1
  return 0
}
main "$@"; exit $?
