#!/bin/bash
# objb-first.sh <log> <sealed .bit path> <its sha256>: the ObjBounds rung on board 2. Flash capstone-ariane objbounds-b
# 603528b8b (bitstream 3 without the per-access lifetime check; rtl's record land/rtl-objb c8be354d28ef), then the
# ObjBounds perf kits, pilot first. The lead's word for this flash, in the runtime session on 2026-10-30: "Я предлагаю
# плату 2 после проверочных загрузок B... — да, делай". Pre-registered in PREREG-t1-t2-sampling.md (ObjBounds section)
# before launch. b4-first.sh's shape, with these changes:
# - the wait is for B's third check boot to end (spec/b4-t1c.log `TC DONE`), then for the sealed .bit to appear at the
#   given path (synth names it; nothing about it is hardcoded but the variant check below), then for the board to be free;
# - the fallback that must stay registered is bitstream 3, caplifive_vp2c_e7af48457.bit;
# - the kit is make-objb-kit.sh's one-line copy of bkit12-perf-b3-b2 (BOARD_BITSTREAM = the sealed name);
# - before any Linux boot, the RTL lane's bare identity kit (objb-bare.sh, ten images) must read CONFIRMED;
# - then boot 1 is the pilot kit objbh1 through this directory's chain-board.sh/spec-board.sh, gated by check-objb-boot.py:
#   VALID arms objbh2..3 and objba1..4 (make-objb-stages.py --arm) and runs the whole list; anything else stops here
#   with the main kits disarmed.
# Each step gates the next. OB_* stub every external step for the dry run.
main() {
  set -u
  local O=${1:?log} BIT=${2:?sealed .bit path} SHA=${3:?sha256 of the sealed .bit} BRD=2
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone} S NAME t0 rc bid r
  S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  NAME=$(basename "$BIT" .bit)
  local ROOT=${OB_ROOT:-$H/perf/objb/board/out} STAGES=${OB_STAGES:-$H/perf/objb/board/stages}
  local KITS=${OB_KITS:-$H/perf/objb/kits} WAITF=${OB_WAITF:-$H/spec/b4-t1c.log}
  local KIT=${OB_KDST:-$H/t3-smoke}/bkit15-objb-b2 B4=$H/wt-spec/capstone/ports/spec-cpu2017/results/2026-10-28-board-test/b4
  export FPGA_BOARD=2 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url-board2
  local BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2"
  local STORE="python3 $H/wt-spec/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/board-tier12/batch/store-has.py"
  case $NAME in caplifive_vp2c_*) ;; *) echo "OB STOP: $NAME is not a caplifive_vp2c_ bitstream" >> "$O"; return 2;; esac
  case $NAME in *e7af48457*|*325966ee4*) echo "OB STOP: $NAME is bitstream 3 or 4, not ObjBounds" >> "$O"; return 2;; esac
  [[ $SHA =~ ^[0-9a-f]{64}$ ]] || { echo "OB STOP: the sha256 argument is not 64 hex digits" >> "$O"; return 2; }
  [ -e "$ROOT" ] && { echo "OB STOP: $ROOT exists" >> "$O"; return 2; }
  echo "OB START $(date '+%F %T') bitstream $NAME sha256 ${SHA:0:16} board $BRD" >> "$O"
  # 1. B's third check boot (b4-t1c) holds board 2 until its log says TC DONE.
  t0=$(date +%s)
  until grep -q "^TC DONE" "$WAITF" 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${OB_WAITMAX:-43200} ] && { echo "OB STOP: b4-t1c did not end" >> "$O"; return 1; }
    sleep ${OB_SLEEP:-60}
  done
  echo "OB b4-t1c ended: $(grep '^TC DONE' "$WAITF" | tail -1)" >> "$O"
  # 2. The sealed bitstream: wait for the file, then its hash.
  t0=$(date +%s)
  until [ -s "$BIT" ]; do
    [ $(( $(date +%s) - t0 )) -gt ${OB_BITMAX:-129600} ] && { echo "OB STOP: $BIT never appeared" >> "$O"; return 1; }
    sleep ${OB_SLEEP:-60}
  done
  sleep ${OB_SETTLE:-30}   # a copy in progress has a size before it has its content
  [ "$(sha256sum "$BIT" | cut -d' ' -f1)" = "$SHA" ] || { echo "OB STOP: local bitstream hash is not $SHA" >> "$O"; return 1; }
  echo "OB bitstream $BIT sha256 matches $(date +%T)" >> "$O"
  # 3. The stages: the pilot armed, every main kit disarmed, every kit verifying on the host.
  [ -f "$STAGES/perf-objb-objbh1/plan.json" ] || { echo "OB STOP: the pilot stage is not armed" >> "$O"; return 2; }
  for k in objbh2 objbh3 objba1 objba2 objba3 objba4; do
    [ -f "$STAGES/perf-objb-$k/plan.json.disarmed" ] && [ ! -e "$STAGES/perf-objb-$k/plan.json" ] \
      || { echo "OB STOP: main stage $k is not disarmed" >> "$O"; return 2; }
  done
  for k in heap15/objbh1 heap15/objbh2 heap15/objbh3 other88/objba1 other88/objba2 other88/objba3 other88/objba4; do
    ( cd "$KITS/$k/stage" && sha256sum -c --quiet ../SHA256SUMS ) || { echo "OB STOP: kit $k does not verify" >> "$O"; return 2; }
  done
  # 4. The board.
  t0=$(date +%s)
  until ${OB_BUSY:-$BUSY} > /dev/null; do
    [ $(( $(date +%s) - t0 )) -gt ${OB_BUSYMAX:-14400} ] && { echo "OB STOP: board 2 stayed busy" >> "$O"; return 1; }
    sleep ${OB_SLEEP:-60}
  done
  echo "OB board 2 free $(date +%T)" >> "$O"
  ${OB_STORE:-$STORE} no-such-bitstream-$$.bit >> "$O" 2>&1; rc=$?
  [ $rc = 1 ] || { echo "OB STOP: store-has control read rc=$rc, not 1" >> "$O"; return 1; }
  ${OB_STORE:-$STORE} caplifive_vp2c_e7af48457.bit >> "$O" 2>&1 || { echo "OB STOP: the fallback (bitstream 3) is not registered" >> "$O"; return 1; }
  # 5. The flash, and the resident record re-read on a fresh session.
  bid=$(${OB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 30 "vp2 ObjBounds $NAME REFLASH (board $BRD, the lead's word)"); rc=$?
  [ $rc = 3 ] && { echo "OB STOP: the hub refused the flash booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "OB NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "OB FLASH BEGIN $(date +%T) board $BRD booking ${bid:-none}" >> "$O"
  ${OB_FLASH:-python3 $H/wt-spec/capstone/tests/rtl-smoke/drivers/flash-bitstream.py} "$NAME.bit" "$BIT" "$SHA" >> "$O" 2>&1; rc=$?
  [ -n "$bid" ] && ${OB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook "$bid" >> "$O" 2>&1
  echo "OB FLASH END rc=$rc $(date +%T)" >> "$O"
  [ $rc = 0 ] || { echo "OB STOP: the flash did not read back $NAME.bit" >> "$O"; return 1; }
  ${OB_NVRB:-python3 $B4/nv-readback.py} "$NAME.bit" "$SHA" >> "$O" 2>&1; rc=$?
  case $rc in
    0) echo "OB READBACK name and sha256 match" >> "$O";;
    3) echo "OB READBACK name matches; the console exposes no sha256" >> "$O";;
    *) echo "OB STOP: readback rc=$rc" >> "$O"; return 1;;
  esac
  # 5b. The RTL lane's bare identity kit (land/rtl-objb beb5384abb3b, Addendum 1 177c6fd9f6e9), before any Linux boot:
  # objb-stale and objb-pcc must read the ObjBounds vectors. Only CONFIRMED (exit 0) goes on; VOID 1, STOP 3 (board 2
  # reads bitstream 3's objb-stale vector) and FINDING 4 stop the driver here, with the grade in the log.
  echo "OB BARE BEGIN $(date +%T)" >> "$O"
  ${OB_BARE:-bash $S/objb-bare.sh} $BRD "$NAME" "$ROOT/bare" >> "$O" 2>&1; r=$?
  sed 's/^/  /' "$ROOT/bare/grade.txt" >> "$O" 2>/dev/null
  echo "OB BARE END rc=$r $(date +%T)" >> "$O"
  case $r in
    0) echo "OB BARE CONFIRMED: the variant's identity reads on board 2" >> "$O";;
    1) echo "OB STOP: bare kit VOID (the control failed or an image did not read its SR)" >> "$O"; touch "$ROOT/.objb-stopped"; return 1;;
    3) echo "OB STOP: bare kit STOP: board 2 reads bitstream 3's objb-stale vector; the variant is not on the board" >> "$O"; touch "$ROOT/.objb-stopped"; return 3;;
    4) echo "OB STOP: bare kit FINDING (identity not confirmed); the reading goes to the RTL lane" >> "$O"; touch "$ROOT/.objb-stopped"; return 4;;
    *) echo "OB STOP: bare kit not run or no data (rc=$r)" >> "$O"; touch "$ROOT/.objb-stopped"; return 1;;
  esac
  # 6. The kit.
  ${OB_KIT:-bash $S/make-objb-kit.sh} ${OB_KSRC:-$H/t3-smoke}/bkit12-perf-b3-b2 "$KIT" "$NAME" "$O" || { echo "OB STOP: kit" >> "$O"; return 1; }
  export SB_KIT=$KIT SB_BITNAME=$NAME
  # 7. The list, written here with the paths expanded (chain-board.sh reads literal paths).
  mkdir -p "$ROOT"
  { echo "# ObjBounds on board 2 ($NAME): stage step_s booking_min twin. The pilot first; the main kits are disarmed until its gate passes."
    echo "$STAGES/perf-objb-objbh1 3600 90 -"
    echo "$STAGES/perf-objb-objbh2 3600 90 -"
    echo "$STAGES/perf-objb-objbh3 3600 90 -"
    for k in objba1 objba2 objba3 objba4; do echo "$STAGES/perf-objb-$k 10800 210 -"; done
  } > "$ROOT/objb-b2.list"
  grep -v '^#' "$ROOT/objb-b2.list" | head -1 > "$ROOT/pilot.list"
  # 8. Boot 1: the pilot, then its gate.
  echo "OB PILOT BEGIN $(date +%T)" >> "$O"
  ${OB_CHAIN:-bash $S/chain-board.sh} $BRD "$ROOT/pilot.list" "$ROOT" >> "$O" 2>&1; echo "OB PILOT END rc=$? $(date +%T)" >> "$O"
  sed 's/^/  /' "$ROOT/chain-b$BRD.log" >> "$O" 2>/dev/null
  ${OB_GATE:-python3 $S/check-objb-boot.py} "$ROOT/perf-objb-objbh1" "$KITS/heap15/objbh1" > "$ROOT/pilot-gate.txt" 2>&1; r=$?
  sed 's/^/  /' "$ROOT/pilot-gate.txt" >> "$O"; echo "OB GATE rc=$r" >> "$O"
  if [ $r != 0 ]; then
    echo "OB HOLD: the pilot is not VALID (1 VOID, 2 NO DATA); the main kits stay disarmed" >> "$O"
    touch "$ROOT/.objb-stopped"; echo "OB DONE $(date +%T)" >> "$O"; return 1
  fi
  ${OB_ARM:-python3 $S/make-objb-stages.py} --arm --out "$STAGES" >> "$O" 2>&1 || { echo "OB STOP: arming failed" >> "$O"; touch "$ROOT/.objb-stopped"; return 1; }
  # 9. The whole list (chain-board.sh skips the pilot, whose out dir exists).
  echo "OB MAIN BEGIN $(date +%T)" >> "$O"
  ${OB_CHAIN:-bash $S/chain-board.sh} $BRD "$ROOT/objb-b2.list" "$ROOT" >> "$O" 2>&1; r=$?
  echo "OB MAIN END rc=$r $(date +%T)" >> "$O"
  for k in objbh2 objbh3 objba1 objba2 objba3 objba4; do
    g=heap15; case $k in objba*) g=other88;; esac
    ${OB_GATE:-python3 $S/check-objb-boot.py} "$ROOT/perf-objb-$k" "$KITS/$g/$k" 2>&1 | head -4 | sed 's/^/  /' >> "$O"
  done
  [ $r = 0 ] && touch "$ROOT/.objb-done" || touch "$ROOT/.objb-stopped"
  echo "OB DONE $(date +%T)" >> "$O"
  return $r
}
main "$@"; exit $?
