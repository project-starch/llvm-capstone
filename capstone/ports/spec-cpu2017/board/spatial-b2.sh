#!/bin/bash
# spatial-b2.sh: board 2's SPEC train queue for the lead's set Plain / Spatial / Full (2026-11-01 13:30).
# Spatial = ObjBounds = the capstone-spatial (MapLife) stages, unchanged, on the bitstream without the per-access lifetime
# check (caplifive_vp2c_603528b8b), through spec-board.sh with SB_KIT=bkit15-objb-b2 (bkit12-perf-b3-b2 byte for byte
# except BOARD_BITSTREAM). Out root out-objb. Steps:
#  1. wait for board 2's current chain (wrapper pid SP_PID) to end with out-hfl/.chain-b2-done (mcf ObjLife is its last stage);
#  2. host checks, board free, store-has control and fallback, booked flash of ObjBounds, readback by name;
#  3. the RTL lane's bare identity kit, CONFIRMED only (as Caps-v);
#  4. pilot lbm Spatial, gated by spatial-gate.py against lbm MapLife; then the other six;
#  5. on every exit after the flash: board 2 back to bitstream 3 (b2-to-bs3.sh);
#  6. if it is back on bitstream 3: after-train.sh 2, i.e. the ObjLife stages left (out-hfl) then the gated Plain stages
#     (out-pm), shared with board 1 by the skip-if-exists rule.
# SP_* stub every external step for the dry run.
main() {
  set -u
  local H=/mnt/hdd/$USER/capstone S HF O L t0 rc r bid dead
  S=$H/spec/board; HF=${SP_HFL:-$S/out-hfl}; O=${SP_OUT:-$S/out-objb}; L=$O/spatial-b2.log
  local PID=${SP_PID:-2717631} NAME=caplifive_vp2c_603528b8b SHA=0f03b097bbd1693bdc3424a872eb2c88f086d5d5fc3d771b29d4945490ec18c1
  local BIT=${SP_BIT:-$H/bitstreams/$NAME.bit} KIT=${SP_KIT:-$H/t3-smoke/bkit15-objb-b2}
  local CB=${SP_CHAIN:-bash $H/wt-spec/capstone/ports/spec-cpu2017/board/chain-board.sh}
  local BUSY=${SP_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2}
  local STORE=${SP_STORE:-python3 $H/wt-spec/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/board-tier12/batch/store-has.py}
  local BOOK=${SP_BOOKER:-python3 /tmp/capstone/lab-hub/lab-book.py}
  local FLASH=${SP_FLASH:-python3 $H/wt-spec/capstone/tests/rtl-smoke/drivers/flash-bitstream.py}
  local NVRB=${SP_NVRB:-python3 $H/wt-spec/capstone/ports/spec-cpu2017/results/2026-10-28-board-test/b4/nv-readback.py}
  local BARE=${SP_BARE:-bash $H/wt-objb/capstone/benchmarks/virtual/board/objb/objb-bare.sh}
  local GATE=${SP_GATE:-python3 $S/spatial-gate.py} AFTER=${SP_AFTER:-bash $S/after-train.sh}
  local SL=${SP_LIST:-$S/spatial-b2.list}
  mkdir -p "$O"
  export FPGA_BOARD=2 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url-board2
  echo "SP START $(date '+%F %T') waiting for board 2's chain (pid $PID)" >> "$L"
  # 1.
  t0=$(date +%s)
  while kill -0 "$PID" 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${SP_WAITMAX:-172800} ] && { echo "SP STOP: pid $PID still running" >> "$L"; return 1; }
    sleep ${SP_SLEEP:-60}
  done
  [ -e "$HF/.chain-b2-done" ] || { echo "SP STOP: board 2's chain ended without .chain-b2-done; nothing launched" >> "$L"; return 1; }
  echo "SP chain ended $(date +%T)" >> "$L"
  # 2. Host checks before the board is touched.
  [ "$(sha256sum "$BIT" | cut -d' ' -f1)" = "$SHA" ] || { echo "SP STOP: local bitstream hash is not $SHA" >> "$L"; return 1; }
  ( cd "$KIT" && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$NAME" "$KIT/board.env" \
    || { echo "SP STOP: $KIT is not the ObjBounds kit" >> "$L"; return 1; }
  grep -v '^#' "$SL" | while read -r st _; do [ -f "$st/plan.json" ] || echo "missing $st"; done | grep -q . \
    && { echo "SP STOP: a stage in $SL is missing" >> "$L"; return 1; }
  [ -f "$HF/519.lbm_r-train-capstone-spatial/result.json" ] || { echo "SP STOP: no lbm MapLife reference" >> "$L"; return 1; }
  t0=$(date +%s)
  until $BUSY > /dev/null 2>&1; do
    [ $(( $(date +%s) - t0 )) -gt ${SP_BUSYMAX:-14400} ] && { echo "SP STOP: board 2 stayed busy" >> "$L"; return 1; }
    sleep ${SP_SLEEP:-60}
  done
  $STORE no-such-bitstream-$$.bit >> "$L" 2>&1; rc=$?
  [ $rc = 1 ] || { echo "SP STOP: store-has control read rc=$rc, not 1" >> "$L"; return 1; }
  $STORE caplifive_vp2c_e7af48457.bit >> "$L" 2>&1 || { echo "SP STOP: the fallback (bitstream 3) is not registered" >> "$L"; return 1; }
  $STORE $NAME.bit >> "$L" 2>&1 || { echo "SP STOP: $NAME is not registered" >> "$L"; return 1; }
  # From the flash on, every exit goes through back().
  bid=$($BOOK book 2 30 "vp2 ObjBounds $NAME REFLASH for SPEC Spatial (board 2)"); rc=$?
  [ $rc = 3 ] && { echo "SP STOP: the hub refused the flash booking (409): $bid" >> "$L"; return 1; }
  [ $rc = 0 ] || { echo "SP NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$L"; bid=; }
  echo "SP FLASH BEGIN $(date +%T) booking ${bid:-none}" >> "$L"
  $FLASH "$NAME.bit" "$BIT" "$SHA" >> "$L" 2>&1; rc=$?
  [ -n "$bid" ] && $BOOK unbook "$bid" >> "$L" 2>&1
  echo "SP FLASH END rc=$rc $(date +%T)" >> "$L"
  [ $rc = 0 ] || { echo "SP STOP: the flash did not read back $NAME.bit" >> "$L"; back "$L"; return 1; }
  $NVRB "$NAME.bit" "$SHA" >> "$L" 2>&1; rc=$?
  case $rc in 0|3) echo "SP READBACK ok rc=$rc" >> "$L";; *) echo "SP STOP: readback rc=$rc" >> "$L"; back "$L"; return 1;; esac
  # 3.
  echo "SP BARE BEGIN $(date +%T)" >> "$L"
  $BARE 2 "$NAME" "$O/bare" >> "$L" 2>&1; r=$?
  sed 's/^/  /' "$O/bare/grade.txt" >> "$L" 2>/dev/null
  echo "SP BARE END rc=$r $(date +%T)" >> "$L"
  if [ $r = 0 ]; then
    export SB_KIT=$KIT SB_BITSTREAM=603528b8b
    # 4.
    grep -m1 '519.lbm_r-train-capstone-spatial ' "$SL" > "$O/pilot.list"
    echo "SP PILOT BEGIN $(date +%T)" >> "$L"
    $CB 2 "$O/pilot.list" "$O" >> "$L" 2>&1; echo "SP PILOT END rc=$? $(date +%T)" >> "$L"
    $GATE "$O/519.lbm_r-train-capstone-spatial/result.json" "$HF/519.lbm_r-train-capstone-spatial/result.json" > "$O/pilot-gate.txt" 2>&1; r=$?
    sed 's/^/  /' "$O/pilot-gate.txt" >> "$L"; echo "SP GATE rc=$r" >> "$L"
    if [ $r = 0 ]; then
      rm -f "$O/.chain-b2-done" "$O/.chain-b2-stopped"
      echo "SP MAIN BEGIN $(date +%T)" >> "$L"
      $CB 2 "$SL" "$O" >> "$L" 2>&1; r=$?
      echo "SP MAIN END rc=$r $(date +%T)" >> "$L"
    else
      echo "SP HOLD: the pilot did not pass its gate; the other six do not run" >> "$L"
    fi
    unset SB_KIT SB_BITSTREAM
  else
    echo "SP HOLD: bare kit rc=$r (0 CONFIRMED only); no SPEC stage on this flash" >> "$L"
  fi
  # 5.
  back "$L" || { echo "SP STOP: board 2 is not confirmed back on bitstream 3; nothing more launched" >> "$L"; return 1; }
  # 6. after-train.sh waits for a pid: give it one that has exited. .chain-b2-done in out-hfl is still the one from step 1.
  ( : ) & dead=$!; wait $dead
  echo "SP CONTINUE $(date +%T): after-train.sh 2 (ObjLife rest, then Plain)" >> "$L"
  $AFTER 2 "$dead" >> "$L" 2>&1; r=$?
  echo "SP DONE $(date +%T) after-train rc=$r" >> "$L"
}
# back <log>: board 2 to bitstream 3 (board free, fallback, hash, booked flash, readback).
back() {
  local r
  echo "SP BACK BEGIN $(date +%T)" >> "$1"
  ${SP_BACK:-bash /mnt/hdd/$USER/capstone/spec/board/b2-to-bs3.sh} "$1"; r=$?
  echo "SP BACK END rc=$r $(date +%T)" >> "$1"
  return $r
}
main "$@"; exit $?
