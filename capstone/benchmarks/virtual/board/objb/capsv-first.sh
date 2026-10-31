#!/bin/bash
# capsv-first.sh <log> <wait-file> <wait-regex>: Caps-v, the Caps perf kits (T0a, unchanged) on the ObjBounds bitstream
# (capstone-ariane 603528b8b) on board 2, then board 2 back to bitstream 3. Caps-v is "capabilities without the per-access
# lifetime check": coarse heap bounds, no per-object lifetimes, no temporal check. Pre-registered in
# PREREG-t1-t2-sampling.md (section "Caps-v") before launch. A copy of objb-first.sh with these changes:
# - the wait is for <wait-regex> in <wait-file> (board 2's SPEC chain ending its current boot), then for the board;
# - the sealed .bit already exists: only its sha256 is checked;
# - the kit is the existing bkit15-objb-b2 (checked by SHA256SUMS and board.env), not rebuilt;
# - the stages are copies of perf/T0a/board/stages/perf-caps-* in perf/capsv/stages, the gate check-caps-boot.py;
# - after the kits (or any stop after the flash) board 2 is reflashed to bitstream 3 by spec/board/b2-to-bs3.sh.
# The RTL lane's bare identity kit still runs before any Linux boot. CV_* stub every external step for the dry run.
main() {
  set -u
  local O=${1:?log} WF=${2:?wait file} WR=${3:?wait regex} BRD=2
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone} S t0 rc bid r
  S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local NAME=caplifive_vp2c_603528b8b SHA=0f03b097bbd1693bdc3424a872eb2c88f086d5d5fc3d771b29d4945490ec18c1
  local BIT=${CV_BIT:-$H/bitstreams/$NAME.bit}
  local ROOT=${CV_ROOT:-$H/perf/capsv/board/out} STAGES=${CV_STAGES:-$H/perf/capsv/stages} KITS=${CV_KITS:-$H/perf/T0a/kits}
  local KIT=${CV_KIT:-$H/t3-smoke/bkit15-objb-b2} B4=$H/wt-spec/capstone/ports/spec-cpu2017/results/2026-10-28-board-test/b4
  export FPGA_BOARD=2 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url-board2
  local BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2"
  local STORE="python3 $H/wt-spec/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/board-tier12/batch/store-has.py"
  local GATE="python3 $H/perf/T0a/board/check-caps-boot.py"
  [ -e "$ROOT" ] && { echo "CV STOP: $ROOT exists" >> "$O"; return 2; }
  echo "CV START $(date '+%F %T') bitstream $NAME board $BRD" >> "$O"
  # 1. Board 2's SPEC chain ends its current boot.
  t0=$(date +%s)
  until grep -qE "$WR" "$WF" 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${CV_WAITMAX:-86400} ] && { echo "CV STOP: '$WR' never appeared in $WF" >> "$O"; return 1; }
    sleep ${CV_SLEEP:-60}
  done
  echo "CV wait ended: $(grep -E "$WR" "$WF" | tail -1)" >> "$O"
  # 2. Bitstream, kit, stages, kits: all checked on the host before the board is touched.
  [ "$(sha256sum "$BIT" | cut -d' ' -f1)" = "$SHA" ] || { echo "CV STOP: local bitstream hash is not $SHA" >> "$O"; return 1; }
  ( cd "$KIT" && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$NAME" "$KIT/board.env" \
    || { echo "CV STOP: $KIT is not the ObjBounds kit" >> "$O"; return 2; }
  for k in heap15/capsh1 heap15/capsh2 heap15/capsh3 other88/capsa1 other88/capsa2 other88/capsa3 other88/capsa4; do
    [ -f "$STAGES/perf-caps-${k#*/}/plan.json" ] || { echo "CV STOP: stage ${k#*/} missing" >> "$O"; return 2; }
    ( cd "$KITS/$k/stage" && sha256sum -c --quiet ../SHA256SUMS ) || { echo "CV STOP: kit $k does not verify" >> "$O"; return 2; }
  done
  # 3. The board.
  t0=$(date +%s)
  until ${CV_BUSY:-$BUSY} > /dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${CV_BUSYMAX:-14400} ] && { echo "CV STOP: board 2 stayed busy" >> "$O"; return 1; }
    sleep ${CV_SLEEP:-60}
  done
  echo "CV board 2 free $(date +%T)" >> "$O"
  ${CV_STORE:-$STORE} no-such-bitstream-$$.bit >> "$O" 2>&1; rc=$?
  [ $rc = 1 ] || { echo "CV STOP: store-has control read rc=$rc, not 1" >> "$O"; return 1; }
  ${CV_STORE:-$STORE} caplifive_vp2c_e7af48457.bit >> "$O" 2>&1 || { echo "CV STOP: the fallback (bitstream 3) is not registered" >> "$O"; return 1; }
  # 4. The flash and its readback. From here on, every exit goes through the reflash to bitstream 3 (back()).
  bid=$(${CV_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 30 "vp2 ObjBounds $NAME REFLASH for Caps-v (board $BRD)"); rc=$?
  [ $rc = 3 ] && { echo "CV STOP: the hub refused the flash booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "CV NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "CV FLASH BEGIN $(date +%T) board $BRD booking ${bid:-none}" >> "$O"
  ${CV_FLASH:-python3 $H/wt-spec/capstone/tests/rtl-smoke/drivers/flash-bitstream.py} "$NAME.bit" "$BIT" "$SHA" >> "$O" 2>&1; rc=$?
  [ -n "$bid" ] && ${CV_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook "$bid" >> "$O" 2>&1
  echo "CV FLASH END rc=$rc $(date +%T)" >> "$O"
  [ $rc = 0 ] || { echo "CV STOP: the flash did not read back $NAME.bit" >> "$O"; back "$O"; return 1; }
  ${CV_NVRB:-python3 $B4/nv-readback.py} "$NAME.bit" "$SHA" >> "$O" 2>&1; rc=$?
  case $rc in
    0|3) echo "CV READBACK ok rc=$rc" >> "$O";;
    *) echo "CV STOP: readback rc=$rc" >> "$O"; back "$O"; return 1;;
  esac
  # 5. The RTL lane's bare identity kit: only CONFIRMED goes on.
  mkdir -p "$ROOT"
  echo "CV BARE BEGIN $(date +%T)" >> "$O"
  ${CV_BARE:-bash $S/objb-bare.sh} $BRD "$NAME" "$ROOT/bare" >> "$O" 2>&1; r=$?
  sed 's/^/  /' "$ROOT/bare/grade.txt" >> "$O" 2>/dev/null
  echo "CV BARE END rc=$r $(date +%T)" >> "$O"
  [ $r = 0 ] || { echo "CV STOP: bare kit rc=$r (0 CONFIRMED only)" >> "$O"; touch "$ROOT/.capsv-stopped"; back "$O"; return 1; }
  export SB_KIT=$KIT SB_BITNAME=$NAME
  # 6. The pilot (capsh1) and its gate, then the rest.
  { echo "# Caps-v on board 2 ($NAME): stage step_s booking_min twin; the step bounds are the Caps run's."
    for k in capsh1 capsh2 capsh3; do echo "$STAGES/perf-caps-$k 3600 90 -"; done
    for k in capsa1 capsa2 capsa3 capsa4; do echo "$STAGES/perf-caps-$k 10800 210 -"; done
  } > "$ROOT/capsv-b2.list"
  grep -v '^#' "$ROOT/capsv-b2.list" | head -1 > "$ROOT/pilot.list"
  echo "CV PILOT BEGIN $(date +%T)" >> "$O"
  ${CV_CHAIN:-bash $S/chain-board.sh} $BRD "$ROOT/pilot.list" "$ROOT" >> "$O" 2>&1; echo "CV PILOT END rc=$? $(date +%T)" >> "$O"
  ${CV_GATE:-$GATE} "$ROOT/perf-caps-capsh1" "$KITS/heap15/capsh1" > "$ROOT/pilot-gate.txt" 2>&1; r=$?
  sed 's/^/  /' "$ROOT/pilot-gate.txt" >> "$O"; echo "CV GATE rc=$r" >> "$O"
  if [ $r != 0 ]; then
    echo "CV HOLD: the pilot is not VALID; the other six kits do not run" >> "$O"; touch "$ROOT/.capsv-stopped"; back "$O"; return 1
  fi
  echo "CV MAIN BEGIN $(date +%T)" >> "$O"
  ${CV_CHAIN:-bash $S/chain-board.sh} $BRD "$ROOT/capsv-b2.list" "$ROOT" >> "$O" 2>&1; r=$?
  echo "CV MAIN END rc=$r $(date +%T)" >> "$O"
  for k in heap15/capsh2 heap15/capsh3 other88/capsa1 other88/capsa2 other88/capsa3 other88/capsa4; do
    ${CV_GATE:-$GATE} "$ROOT/perf-caps-${k#*/}" "$KITS/$k" 2>&1 | head -3 | sed 's/^/  /' >> "$O"
  done
  [ $r = 0 ] && touch "$ROOT/.capsv-done" || touch "$ROOT/.capsv-stopped"
  back "$O"; r2=$?
  echo "CV DONE $(date +%T) main rc=$r back rc=$r2" >> "$O"
  return $(( r || r2 ))
}
# back <log>: board 2 to bitstream 3; b2-to-bs3.sh checks the board is free, the fallback, the hash, then flashes and reads back.
back() {
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone} r
  echo "CV BACK BEGIN $(date +%T)" >> "$1"
  ${CV_BACK:-bash $H/spec/board/b2-to-bs3.sh} "$1"; r=$?
  echo "CV BACK END rc=$r $(date +%T)" >> "$1"
  return $r
}
main "$@"; exit $?
