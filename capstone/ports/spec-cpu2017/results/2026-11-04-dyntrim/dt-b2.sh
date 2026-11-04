#!/bin/bash
# dt-b2.sh: stage A of the DYN-unit trim (capstone-ariane c9c7952bd, sealed sha256 790ac2c0...) on board 2, after the
# C.5a session there (c5a-spec-b2.sh, pid DT_WAITPID) has ended with DONE and board 2 is back on bitstream 3.
# Pre-registration: results/2026-11-04-dyntrim/PREREG.md (DT_PREREG names its commit; must be on origin).
#  1. Flash caplifive_vp2c_c9c7952bd, read the resident name back.
#  2. R-67 kit (fixed --bitstream c9c7952bd; gates the rest), hpm kit, ObjBounds kit (objb-pcc LAST), slot kit: graded
#     with the dyntrim arms (land/rtl-c5a-kits 210768baa03d, land/rtl-slot-kit 98ff08e6f3fb) against board 2's C.5a runs.
#  3. SPEC test (c5a-spec-b1.list, kit bkit19-perf-dt-b2) into out-dt-b2 only if R-67 rc 0, ObjBounds rc 0, the hpm grade
#     reads neither WRONG BITSTREAM nor NO DATA (its open l1d_miss FINDING is not a blocker), and no HARD STOP.
#  4. Flash bitstream 3 back.
# Stubs for the dry run: DT_BUSY DT_BOOK DT_FLASH DT_READBACK DT_STOREHAS DT_RUN DT_CHAIN DT_O DT_SPECOUT DT_WAITLOG.
main() {
  set -u
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}
  local O=${DT_O:-$H/perf/dt-b2} P=$H/perf L
  L=$O/dt-b2.log
  local DT=caplifive_vp2c_c9c7952bd DTSHA=790ac2c096a0cd7361129572713333410745a4bc3158c581d7a9c25a5c1e97f2
  local B3=caplifive_vp2c_e7af48457 B3SHA=d6ed97b6d341ea058d497838e43558900823c36591c44046d3cfc0b7f55b423b
  local T=$H/wt-spec/capstone/tests/rtl-smoke SB=$H/wt-spec/capstone/ports/spec-cpu2017
  local RUNNER=/tmp/capstone/wt-b2/capstone/tests/rtl-smoke/sup-bare-2026-10-03-ladder/run_sup_bare.py
  local BOOK=${DT_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} FLASH=${DT_FLASH:-python3 $T/drivers/flash-bitstream.py}
  local RB=${DT_READBACK:-python3 $SB/results/2026-10-28-board-test/b4/nv-readback.py}
  local STORE=${DT_STOREHAS:-python3 $T/vp2-kit-2026-10-17/board-tier12/batch/store-has.py}
  local BUSY=${DT_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2}
  local RUN=${DT_RUN:-python3 $RUNNER} CHAIN=${DT_CHAIN:-bash $SB/board/chain-board.sh}
  local HK=$P/hpm/kit-289ad4e5/images RK=$P/hpm/kit-r67-b1dfe406/images OK=$P/objb/bare-kit/images W=$P/objb/witness
  local SK=$P/slot/kit-98ff08e6 G=${DT_G:-$P/c5a-kits-210768ba/grade} KIT=$H/t3-smoke/bkit19-perf-dt-b2
  local SPEC=${DT_SPECOUT:-$H/spec/board/out-dt-b2} r67=x hpm=x objb=x slotg=x spec=skipped hard=0 bid t
  export FPGA_BOARD=2 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url-board2
  log() { echo "DT $(date '+%F %T') $*" >> "$L"; }
  mkdir -p "$O"; log "START pid $$"
  [ "$(sha256sum "$H/bitstreams/$DT.bit" | cut -d' ' -f1)" = $DTSHA ] || { log "STOP: stage A .bit hash"; return 2; }
  [ "$(sha256sum "$H/bitstreams/$B3.bit" | cut -d' ' -f1)" = $B3SHA ] || { log "STOP: bitstream 3 .bit hash"; return 2; }
  ( cd "$HK" && sha256sum -c --quiet SHA256SUMS ) && ( cd "$RK" && sha256sum -c --quiet SHA256SUMS ) && ( cd "$OK" && sha256sum -c --quiet SHA256SUMS ) \
    && ( cd "$SK/images" && sha256sum -c --quiet SHA256SUMS ) || { log "STOP: a kit does not verify"; return 2; }
  [ "$(sha256sum "$RUNNER" | cut -c1-16)" = 102ab84e762cba50 ] || { log "STOP: runner hash"; return 2; }
  ( cd "$KIT" && sha256sum -c --quiet SHA256SUMS ) && grep -q "^BOARD_BITSTREAM=$DT$" "$KIT/board.env" || { log "STOP: stage A perf kit"; return 2; }
  for f in "$G/hpm-grade.py" "$G/objb-grade.py" "$G/r67-grade.py" "$SK/grade/slot-grade.py" "$O/dt-spec-grade.py"; do [ -f "$f" ] || { log "STOP: $f"; return 2; }; done
  for d in hpm objb; do [ -d "$P/c5a-b2/$d" ] || { log "STOP: comparison dir c5a-b2/$d"; return 2; }; done
  [ -n "${DT_PREREG:-}" ] && git -C "$H/wt-spec" branch -r --contains "$DT_PREREG" 2>/dev/null | grep -q origin/spec-silicon \
    || { log "STOP: pre-registration ${DT_PREREG:-unset} is not on origin/spec-silicon"; return 2; }
  log "preconditions ok; waiting for the C.5a session (pid ${DT_WAITPID:-none})"
  t=0; while [ -n "${DT_WAITPID:-}" ] && kill -0 "$DT_WAITPID" 2>/dev/null; do
    [ $t -ge 86400 ] && { log "STOP: the C.5a session still runs"; return 2; }; sleep "${DT_POLL_S:-60}"; t=$((t + ${DT_POLL_S:-60})); done
  grep -q "^CS .* DONE" "${DT_WAITLOG:-$P/c5a-b2/c5a-spec-b2.log}" || { log "STOP: the C.5a session did not end with DONE"; return 2; }
  $BUSY > /dev/null 2>&1 || { log "STOP: board 2 busy"; return 2; }
  log "board 2 free, on bitstream 3"

  flash() {
    local n=$1 s=$2 b r
    $STORE no-such-bitstream-$$.bit >> "$L" 2>&1; r=$?; [ $r = 1 ] || { log "flash $n: store-has control rc=$r"; return 1; }
    b=$($BOOK book 2 30 "vp2 $n REFLASH (board 2)") || b=
    log "FLASH BEGIN $n booking ${b:-none}"; $FLASH "$n.bit" "$H/bitstreams/$n.bit" "$s" >> "$L" 2>&1; r=$?
    [ -n "$b" ] && $BOOK unbook "$b" >> "$L" 2>&1; log "FLASH END $n rc=$r"; [ $r = 0 ] || return 1
    $RB "$n.bit" "$s" >> "$L" 2>&1; r=$?; case $r in 0|3) log "READBACK $n ok rc=$r";; *) log "READBACK $n rc=$r"; return 1;; esac
  }
  bare() {  # bare <out dir> <image>...
    local out=$1 i d rc; shift; mkdir -p "$out"
    for i in "$@"; do
      case $i in hpm-*) d=$HK;; t1-*) d=$RK;; objb-stale-w) d=$W;; slot-*) d=$SK/images;; *) d=$OK;; esac
      ( export FPGA_URL; FPGA_URL=$(cat "$CAPSTONE_FPGA_URL_FILE")
        FPGA_BITSTREAM=$DT.bit SUP_IMG=$d/$i.bin SUP_REC_ADDR=0x$(sed "s/^0x//" "$d/$i.rec") SUP_OUT=$out/$i \
          SUP_TIMEOUT=90 timeout 900 $RUN > "$out/$i.stderr" 2>&1 ); rc=$?
      echo $rc > "$out/$i.rc"; log "IMAGE $i rc=$rc"
      grep -q 'HARD STOP' "$out/$i.stderr" "$out/$i/run.log" 2>/dev/null && { log "HARD STOP at $i: resident is not $DT"; return 1; }
    done; return 0
  }
  if flash $DT $DTSHA; then
    bid=; for t in $(seq 1 30); do bid=$($BOOK book 2 120 "vp2 stage A first boots: R-67, hpm, ObjBounds, slot kits (board 2)") && break; bid=; sleep 60; done
    if bare "$O/r67" t1-norevoke t1-untagged t1-oob t1-fire; then
      python3 "$G/r67-grade.py" "$O/r67" fixed --bitstream c9c7952bd > "$O/grade-r67.txt" 2>&1; r67=$?
      log "R67 GRADE rc=$r67"; sed 's/^/  /' "$O/grade-r67.txt" >> "$L"
      if [ $r67 = 0 ]; then
        bare "$O/hpm" hpm-5 hpm-3 hpm-4 hpm-6 hpm-1 hpm-2 hpm-7 \
          && bare "$O/slot" slot-9 slot-6 slot-5 slot-7 slot-8 slot-1 slot-2 slot-3 slot-4 \
          && bare "$O/objb" objb-pcc-ctrl objb-stale objb-stale-w objb-perf-s1 objb-perf-s2 objb-perf-s3 objb-perf-s4 \
                            objb-perf-s5 objb-perf-s6 objb-perf-s7 objb-pcc || hard=1
        python3 "$G/hpm-grade.py" "$O/hpm" dyntrim "$P/c5a-b2/hpm" > "$O/grade-hpm.txt" 2>&1; hpm=$?
        log "HPM GRADE rc=$hpm"; sed 's/^/  /' "$O/grade-hpm.txt" >> "$L"
        python3 "$SK/grade/slot-grade.py" "$O/slot" dyntrim > "$O/grade-slot.txt" 2>&1; slotg=$?
        log "SLOT GRADE rc=$slotg"; sed 's/^/  /' "$O/grade-slot.txt" >> "$L"
        python3 "$G/objb-grade.py" "$O/objb" dyntrim "$P/c5a-b2/objb" > "$O/grade-objb.txt" 2>&1; objb=$?
        log "OBJB GRADE rc=$objb"; sed 's/^/  /' "$O/grade-objb.txt" >> "$L"
      else log "R-67 not FIXED on stage A: nothing else runs on it (it goes to the lead)"; fi
    fi
    [ -n "$bid" ] && $BOOK unbook "$bid" >> "$L" 2>&1
    if [ "$r67$objb$hard" = 000 ] && ! grep -q "WRONG BITSTREAM\|NO DATA" "$O/grade-hpm.txt"; then
      log "SPEC on stage A BEGIN"
      SB_KIT=$KIT SB_BITSTREAM=c9c7952bd $CHAIN 2 "$O/c5a-spec-b1.list" "$SPEC" >> "$L" 2>&1; spec=$?
      log "SPEC on stage A END chain rc=$spec"
      CAPSTONE_WORK=$H DT_OUT=$SPEC python3 "$O/dt-spec-grade.py" > "$O/grade-spec.txt" 2>&1; log "SPEC GRADE rc=$?"; sed 's/^/  /' "$O/grade-spec.txt" >> "$L"
    else log "SPEC on stage A skipped (r67 $r67 hpm $hpm objb $objb hard-stop $hard)"; fi
  else log "stage A flash failed"; fi
  flash $B3 $B3SHA || { log "STOP: the flash back to bitstream 3 failed; board 2 left as is"; return 1; }
  log "DONE r67 $r67 hpm $hpm slot $slotg objb $objb spec $spec; board 2 back on bitstream 3"
}
main "$@"; exit $?
