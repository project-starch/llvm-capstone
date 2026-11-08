#!/bin/bash
# e5n-gate.sh <board 1|2|3> : e5-native's 32-image first-boot gate (rtl lane; copied from perf/e0gate/e0-gate.sh with the
# bitstream, vectors and grader swapped; variable names kept, so "E0" below means the variant under test). Flashes
# caplifive_e5native_14b0a57df (sha256 read from e5n-bit.sha, written at the seal; absent = STOP), reads the
# resident name back, runs the 32 C.5a-gate images one boot each (R-67, hpm, ObjBounds without objb-pcc, slot, R-32,
# objb-pcc LAST since it never reaches its recorder), grades them against the E0 column and C.5a's silicon runs.
# Exit 0 = PASS and the board is LEFT ON E0 (the SPEC stages follow); 1 = FAIL, the board is flashed BACK to C.5a;
# 2 = a precondition failed, nothing was flashed; 3 = a flash/readback failed.
# Stubs for the dry run (as c5a-b2.sh): E0_BUSY E0_BOOK E0_FLASH E0_READBACK E0_STOREHAS E0_RUN E0_O E0_STORE_DIR.
main() {
  set -u
  local B=${1:?board number}
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone} P=/mnt/hdd/$USER/capstone/perf
  local O=${E0_O:-$P/e5ngate/run-b$B-$(date +%m%d-%H%M)} G=$P/e5ngate
  local L=$O/e0-gate.log
  local E0=caplifive_e5native_14b0a57df E0SHA=$(cat ${E0_BITSHA_FILE:-$P/e5ngate/e5n-bit.sha} 2>/dev/null || echo NO-SEAL-YET)
  local C5=caplifive_vp2c_1fc2cd5aa C5SHA=5b7b1edcd9bcd2521bd4bd59107747e268487056f1bfcd96403fd612eeb6d764
  local BS=${E0_STORE_DIR:-$H/bitstreams}
  local T=$H/wt-spec/capstone/tests/rtl-smoke SB=$H/wt-spec/capstone/ports/spec-cpu2017
  local RUNNER=/tmp/capstone/wt-b2/capstone/tests/rtl-smoke/sup-bare-2026-10-03-ladder/run_sup_bare.py
  local BOOK=${E0_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py}
  local FLASH=${E0_FLASH:-python3 $T/drivers/flash-bitstream.py}
  local RB=${E0_READBACK:-python3 $SB/results/2026-10-28-board-test/b4/nv-readback.py}
  local STORE=${E0_STOREHAS:-python3 $T/vp2-kit-2026-10-17/board-tier12/batch/store-has.py}
  local BUSY=${E0_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py $B}
  local RUN=${E0_RUN:-python3 $RUNNER}
  local HK=$P/hpm/kit-289ad4e5/images RK=$P/hpm/kit-r67-b1dfe406/images OK=$P/objb/bare-kit/images W=$P/objb/witness
  local SK=$P/slot/kit-e1a835aa/images RRK=$P/r32/kit-62492457/images bid t0 rc g
  case $B in 1) U=$HOME/.claude-kisp/secrets/fpga-console-url;; 2|3) U=$HOME/.claude-kisp/secrets/fpga-console-url-board$B;;
    *) echo "board must be 1, 2 or 3"; return 2;; esac
  export FPGA_BOARD=$B CAPSTONE_FPGA_URL_FILE=$U
  mkdir -p "$O"; log() { echo "E5N $(date '+%F %T') $*" | tee -a "$L"; }
  log "START board $B pid $$ out $O"

  # --- preconditions, before anything touches the board ---
  [ -f "$U" ] || { log "STOP: no console URL file for board $B"; return 2; }
  [ "$(sha256sum "$BS/$E0.bit" 2>/dev/null | cut -d' ' -f1)" = $E0SHA ] || { log "STOP: E0 .bit missing or hash differs ($BS/$E0.bit)"; return 2; }
  [ "$(sha256sum "$BS/$C5.bit" 2>/dev/null | cut -d' ' -f1)" = $C5SHA ] || { log "STOP: C.5a .bit (the fallback) missing or hash differs"; return 2; }
  for d in "$HK" "$RK" "$OK" "$W" "$SK" "$RRK"; do ( cd "$d" && sha256sum -c --quiet SHA256SUMS ) || { log "STOP: kit $d fails its SHA256SUMS"; return 2; }; done
  [ "$(sha256sum "$RUNNER" | cut -c1-16)" = 102ab84e762cba50 ] || { log "STOP: runner hash"; return 2; }
  [ "$(sha256sum "$G/e5n-vectors.txt" | cut -c1-16)" = "$(cut -c1-16 "$G/e5n-vectors.sha")" ] || { log "STOP: e5n-vectors.txt differs from the pre-registered copy"; return 2; }
  $STORE no-such-bitstream-$$.bit >> "$L" 2>&1; rc=$?; [ $rc = 1 ] || { log "STOP: store-has control rc=$rc, not 1"; return 2; }
  $BUSY > /dev/null 2>&1 || { log "STOP: board $B is busy"; return 2; }
  log "preconditions ok"

  flash() {  # flash <name> <sha>
    local n=$1 s=$2 b r
    b=$($BOOK book $B 30 "E0 gate: $n REFLASH (board $B)"); r=$?; [ $r = 0 ] || { log "flash $n: booking rc=$r; unbooked"; b=; }
    log "FLASH BEGIN $n"; $FLASH "$n.bit" "$BS/$n.bit" "$s" >> "$L" 2>&1; r=$?
    [ -n "$b" ] && $BOOK unbook "$b" >> "$L" 2>&1
    log "FLASH END $n rc=$r"; [ $r = 0 ] || return 1
    $RB "$n.bit" "$s" >> "$L" 2>&1; r=$?
    case $r in 0|3) log "READBACK $n ok rc=$r";; *) log "READBACK $n rc=$r"; return 1;; esac
  }
  img() {  # img <dir> <image>: one boot through run_sup_bare.py
    local d=$1 i=$2
    log "IMAGE $i BEGIN"
    ( export FPGA_URL; FPGA_URL=$(cat "$U")
      FPGA_BITSTREAM=$E0.bit SUP_IMG=$d/$i.bin SUP_REC_ADDR=0x$(sed "s/^0x//" "$d/$i.rec") SUP_OUT=$O/$i \
        SUP_TIMEOUT=90 timeout 900 $RUN > "$O/$i.stderr" 2>&1 ); rc=$?
    echo $rc > "$O/$i.rc"; log "IMAGE $i END rc=$rc"
    grep -q 'HARD STOP' "$O/$i.stderr" "$O/$i/run.log" 2>/dev/null && { log "HARD STOP at $i: resident is not $E0"; return 1; }
    return 0
  }

  flash $E0 $E0SHA || { log "E0 flash failed"; flash $C5 $C5SHA || log "the flash back to C.5a FAILED too: board $B left as is"; return 3; }
  bid=; for t0 in $(seq 1 30); do bid=$($BOOK book $B 120 "E0 gate: 32 bare images (board $B)") && break; bid=; sleep 60; done
  local hard=0
  for i in t1-norevoke t1-untagged t1-oob t1-fire; do img "$RK" $i || { hard=1; break; }; done
  [ $hard = 0 ] && for i in hpm-5 hpm-3 hpm-4 hpm-6 hpm-1 hpm-2 hpm-7; do img "$HK" $i || { hard=1; break; }; done
  [ $hard = 0 ] && for i in objb-pcc-ctrl objb-stale objb-perf-s1 objb-perf-s2 objb-perf-s3 objb-perf-s4 objb-perf-s5 objb-perf-s6 objb-perf-s7; do img "$OK" $i || { hard=1; break; }; done
  [ $hard = 0 ] && { img "$W" objb-stale-w || hard=1; }
  [ $hard = 0 ] && for i in slot-9 slot-6 slot-5 slot-7 slot-8 slot-1 slot-2 slot-3 slot-4; do img "$SK" $i || { hard=1; break; }; done
  [ $hard = 0 ] && { img "$RRK" shrinkto || hard=1; }
  [ $hard = 0 ] && { img "$OK" objb-pcc || hard=1; }
  [ -n "$bid" ] && $BOOK unbook "$bid" >> "$L" 2>&1
  python3 "$G/e5n-grade.py" "$O" "$G/e5n-vectors.txt" > "$O/grade.txt" 2>&1; g=$?
  sed 's/^/  /' "$O/grade.txt" >> "$L"; log "GRADE rc=$g hard-stop $hard"
  if [ $g = 0 ] && [ $hard = 0 ]; then log "E0 GATE PASS: board $B stays on $E0"; return 0; fi
  log "E0 GATE FAIL (grade rc $g, hard-stop $hard): flashing back to C.5a"
  flash $C5 $C5SHA || { log "the flash back to C.5a FAILED: board $B left as is"; return 3; }
  log "board $B back on $C5"; return 1
}
main "$@"; exit $?
