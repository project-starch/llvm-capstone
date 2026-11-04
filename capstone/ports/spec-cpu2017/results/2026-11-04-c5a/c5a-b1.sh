#!/bin/bash
# c5a-b1.sh: the C.5a bitstream's first boots on board 1, then SPEC test on it, then back to bitstream 3 and the SPEC chain.
# Pre-registration: spec-silicon results/2026-11-04-c5a/PREREG.md (C5_PREREG names its commit; it must be on origin).
#  1. Wait until board 1's chain has stopped at the sentinel line `.../STOP-for-c5a-flash` (inserted after mcf-devmk).
#  2. Flash caplifive_vp2c_1fc2cd5aa (sealed sha256 5b7b1edc...), read the resident name back.
#  3. R-67 kit (graded `fixed --bitstream 1fc2cd5aa`). Only if FIXED (rc 0): the hpm kit, then the ObjBounds kit with the
#     witness objb-stale-w, objb-pcc LAST (it does not return). Graded against bitstream 3's runs.
#  4. Only if all three grade rc 0: SPEC test on C.5a (chain-board.sh over c5a-spec-b1.list into spec/board/out-c5a, kit
#     bkit17-perf-c5a-b1), graded by c5a-spec-grade.py.
#  5. Always (after any of the above, or a failure in it): flash bitstream 3 back, read it back, and restart the board 1
#     chain on the main list's lines after the sentinel. A failure of that flash stops here, and the board stays off.
# Stubs for the dry run: C5_BUSY C5_BOOK C5_FLASH C5_READBACK C5_STOREHAS C5_RUN C5_CHAIN C5_CHAINPID C5_POLL_S C5_O,
# and the paths C5_MAINLIST C5_CHROOT C5_SPECOUT C5_REST.
main() {
  set -u
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}
  local O=${C5_O:-$H/perf/c5a-first} P=$H/perf
  local L=$O/c5a-b1.log
  local C5=caplifive_vp2c_1fc2cd5aa C5SHA=5b7b1edcd9bcd2521bd4bd59107747e268487056f1bfcd96403fd612eeb6d764
  local B3=caplifive_vp2c_e7af48457 B3SHA=d6ed97b6d341ea058d497838e43558900823c36591c44046d3cfc0b7f55b423b
  local T=$H/wt-spec/capstone/tests/rtl-smoke SB=$H/wt-spec/capstone/ports/spec-cpu2017
  local MAINLIST=${C5_MAINLIST:-$H/spec/board/b1-resume-oom.list} CHROOT=${C5_CHROOT:-$H/spec/board/out-hfl} SENT=STOP-for-c5a-flash
  local RUNNER=/tmp/capstone/wt-b2/capstone/tests/rtl-smoke/sup-bare-2026-10-03-ladder/run_sup_bare.py
  local BOOK=${C5_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py}
  local FLASH=${C5_FLASH:-python3 $T/drivers/flash-bitstream.py}
  local RB=${C5_READBACK:-python3 $SB/results/2026-10-28-board-test/b4/nv-readback.py}
  local STORE=${C5_STOREHAS:-python3 $T/vp2-kit-2026-10-17/board-tier12/batch/store-has.py}
  local BUSY=${C5_BUSY:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1}
  local RUN=${C5_RUN:-python3 $RUNNER} CHAIN=${C5_CHAIN:-bash $SB/board/chain-board.sh}
  local HK=$P/hpm/kit-289ad4e5/images RK=$P/hpm/kit-r67-b1dfe406/images OK=$P/objb/bare-kit/images W=$P/objb/witness
  local G=$O/grade KIT=$H/t3-smoke/bkit17-perf-c5a-b1 rc r67 hpm objb spec=skipped bid t0 hard=0
  export FPGA_BOARD=1 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url
  log() { echo "C5 $(date '+%F %T') $*" >> "$L"; }
  mkdir -p "$O"; log "START pid $$"

  # --- preconditions, all before the wait ---
  [ "$(sha256sum "$H/bitstreams/$C5.bit" | cut -d' ' -f1)" = $C5SHA ] || { log "STOP: C.5a .bit hash"; return 2; }
  [ "$(sha256sum "$H/bitstreams/$B3.bit" | cut -d' ' -f1)" = $B3SHA ] || { log "STOP: bitstream 3 .bit hash (the fallback)"; return 2; }
  ( cd "$HK" && sha256sum -c --quiet SHA256SUMS ) && [ "$(sha256sum "$HK/SHA256SUMS" | cut -c1-16)" = ccee9644b09f270e ] || { log "STOP: hpm kit"; return 2; }
  ( cd "$RK" && sha256sum -c --quiet SHA256SUMS ) || { log "STOP: R-67 kit"; return 2; }
  ( cd "$OK" && sha256sum -c --quiet SHA256SUMS ) && [ "$(sha256sum "$OK/SHA256SUMS" | cut -c1-16)" = 8dd2a4812c9da608 ] || { log "STOP: ObjBounds kit"; return 2; }
  [ "$(sha256sum "$W/objb-stale-w.bin" | cut -c1-16)" = ae0e1cb35dcc42d3 ] || { log "STOP: witness image"; return 2; }
  [ "$(sha256sum "$RUNNER" | cut -c1-16)" = 102ab84e762cba50 ] || { log "STOP: runner hash"; return 2; }
  ( cd "$KIT" && sha256sum -c --quiet SHA256SUMS ) && grep -q "^BOARD_BITSTREAM=$C5$" "$KIT/board.env" || { log "STOP: C.5a perf kit"; return 2; }
  for f in hpm-grade.py objb-grade.py r67-grade.py; do [ -f "$G/$f" ] || { log "STOP: grader $f"; return 2; }; done
  [ -n "${C5_PREREG:-}" ] && git -C "$H/wt-spec" branch -r --contains "$C5_PREREG" 2>/dev/null | grep -q origin/spec-silicon \
    || { log "STOP: pre-registration ${C5_PREREG:-unset} is not on origin/spec-silicon"; return 2; }
  [ "$(grep -c "/$SENT " "$MAINLIST")" = 1 ] || { log "STOP: the sentinel is not in $MAINLIST exactly once"; return 2; }
  log "preconditions ok; waiting for board 1's chain to stop at the sentinel"

  # --- 1. wait for the chain ---
  local cpid=${C5_CHAINPID:-3629844} waited=0
  until ! kill -0 "$cpid" 2>/dev/null; do
    [ $waited -ge "${C5_WAIT_S:-72000}" ] && { log "STOP: chain pid $cpid still running after $waited s"; return 2; }
    sleep "${C5_POLL_S:-60}"; waited=$((waited + ${C5_POLL_S:-60}))
  done
  tail -3 "$CHROOT/chain-b1.log" | grep -q "CB STOP after $SENT" || { log "STOP: the chain ended, but not at the sentinel; nothing flashed"; tail -5 "$CHROOT/chain-b1.log" >> "$L"; return 2; }
  log "chain stopped at the sentinel"

  flash() {  # flash <name> <sha>
    local n=$1 s=$2 b r i
    for i in $(seq 1 60); do $BUSY > /dev/null 2>&1 && break; sleep 60; done
    $BUSY > /dev/null 2>&1 || { log "flash $n: board busy for an hour"; return 1; }
    $STORE no-such-bitstream-$$.bit >> "$L" 2>&1; r=$?
    [ $r = 1 ] || { log "flash $n: store-has control rc=$r, not 1"; return 1; }
    b=$($BOOK book 1 30 "vp2 $n REFLASH (board 1)"); r=$?
    [ $r = 0 ] || { log "flash $n: booking rc=$r; continuing unbooked"; b=; }
    log "FLASH BEGIN $n booking ${b:-none}"
    $FLASH "$n.bit" "$H/bitstreams/$n.bit" "$s" >> "$L" 2>&1; r=$?
    [ -n "$b" ] && $BOOK unbook "$b" >> "$L" 2>&1
    log "FLASH END $n rc=$r"; [ $r = 0 ] || return 1
    $RB "$n.bit" "$s" >> "$L" 2>&1; r=$?
    case $r in 0|3) log "READBACK $n ok rc=$r";; *) log "READBACK $n rc=$r"; return 1;; esac
  }
  bare() {  # bare <out dir> <image>...: one boot per image through run_sup_bare.py, on C.5a
    local out=$1 i d; shift; mkdir -p "$out"
    for i in "$@"; do
      case $i in hpm-*) d=$HK;; t1-*) d=$RK;; objb-stale-w) d=$W;; *) d=$OK;; esac
      log "IMAGE $i BEGIN"
      ( export FPGA_URL; FPGA_URL=$(cat "$CAPSTONE_FPGA_URL_FILE")
        FPGA_BITSTREAM=$C5.bit SUP_IMG=$d/$i.bin SUP_REC_ADDR=0x$(sed "s/^0x//" "$d/$i.rec") SUP_OUT=$out/$i \
          SUP_TIMEOUT=90 timeout 900 $RUN > "$out/$i.stderr" 2>&1 ); rc=$?
      echo $rc > "$out/$i.rc"; log "IMAGE $i END rc=$rc"
      grep -q 'HARD STOP' "$out/$i.stderr" "$out/$i/run.log" 2>/dev/null && { log "HARD STOP at $i: resident is not $C5"; return 1; }
    done; return 0
  }

  # --- 2-4 ---
  r67=x; hpm=x; objb=x
  if flash $C5 $C5SHA; then
    bid=; for t0 in $(seq 1 30); do bid=$($BOOK book 1 90 "vp2 C.5a first boots: R-67, hpm, ObjBounds kits (board 1)") && break; bid=; sleep 60; done
    log "bare kits booking ${bid:-none}"
    if bare "$O/r67" t1-norevoke t1-untagged t1-oob t1-fire; then
      python3 "$G/r67-grade.py" "$O/r67" fixed --bitstream 1fc2cd5aa > "$O/grade-r67.txt" 2>&1; r67=$?
      log "R67 GRADE rc=$r67"; sed 's/^/  /' "$O/grade-r67.txt" >> "$L"
      if [ $r67 = 0 ]; then
        bare "$O/hpm" hpm-5 hpm-3 hpm-4 hpm-6 hpm-1 hpm-2 hpm-7 \
          && bare "$O/objb" objb-pcc-ctrl objb-stale objb-stale-w objb-perf-s1 objb-perf-s2 objb-perf-s3 objb-perf-s4 \
                            objb-perf-s5 objb-perf-s6 objb-perf-s7 objb-pcc || hard=1
        python3 "$G/hpm-grade.py" "$O/hpm" c5a "$P/hpm/run-b1-bs3" > "$O/grade-hpm.txt" 2>&1; hpm=$?
        log "HPM GRADE rc=$hpm"; sed 's/^/  /' "$O/grade-hpm.txt" >> "$L"
        python3 "$G/objb-grade.py" "$O/objb" c5a "$P/objb/bare-bs3-b2" > "$O/grade-objb.txt" 2>&1; objb=$?
        log "OBJB GRADE rc=$objb"; sed 's/^/  /' "$O/grade-objb.txt" >> "$L"
      else log "R-67 not FIXED: nothing else runs on C.5a (it goes to the lead)"; fi
    fi
    [ -n "$bid" ] && $BOOK unbook "$bid" >> "$L" 2>&1
    if [ "$r67$hpm$objb$hard" = 0000 ]; then
      log "SPEC on C.5a BEGIN"
      SB_KIT=$KIT SB_BITSTREAM=1fc2cd5aa $CHAIN 1 "$O/c5a-spec-b1.list" "${C5_SPECOUT:-$H/spec/board/out-c5a}" >> "$L" 2>&1; spec=$?
      log "SPEC on C.5a END chain rc=$spec"
      CAPSTONE_WORK=$H C5_OUT=${C5_SPECOUT:-$H/spec/board/out-c5a} python3 "$O/c5a-spec-grade.py" > "$O/grade-spec.txt" 2>&1; log "SPEC GRADE rc=$?"; sed 's/^/  /' "$O/grade-spec.txt" >> "$L"
    else log "SPEC on C.5a skipped (r67 $r67 hpm $hpm objb $objb hard-stop $hard)"; fi
  else log "C.5a flash failed"; fi

  # --- 5. back to bitstream 3, then the chain ---
  flash $B3 $B3SHA || { log "STOP: the flash back to bitstream 3 failed; board 1 left as is, chain NOT restarted"; return 1; }
  local rest=${C5_REST:-$H/spec/board/b1-after-c5a.list}
  awk -v s="/$SENT " 'f{print} index($0,s){f=1}' "$MAINLIST" > "$rest"
  log "restarting board 1's chain on $(grep -c -v '^#' "$rest") lines of $rest"
  setsid nohup $CHAIN 1 "$rest" "$CHROOT" > "$O/chain-restart.nohup" 2>&1 < /dev/null &
  log "DONE r67 $r67 hpm $hpm objb $objb spec $spec; chain pid $!"
}
main "$@"; exit $?
