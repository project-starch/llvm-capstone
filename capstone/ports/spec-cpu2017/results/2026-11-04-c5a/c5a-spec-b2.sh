#!/bin/bash
# c5a-spec-b2.sh: SPEC test on C.5a, board 2, after c5a-b2.sh's kit session (C.5a PREREG Addendum 5). Waits for
# c5a-b2.sh (pid C5S_PID) to end; flashes C.5a (sealed sha256, read back), runs c5a-spec-b1.list on kit
# bkit17-perf-c5a-b2 into out-c5a-b2, grades S1-S3 with c5a-spec-grade.py, flashes bitstream 3 back.
# Runs only if c5a-b2.sh's R-67 grade read FIXED, its ObjBounds grade rc 0, its hpm grade neither WRONG BITSTREAM nor
# NO DATA (the one-L1D-miss FINDING is not a blocker, Addendum 5), no HARD STOP, and it did not run SPEC itself.
main() {
  set -u
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}
  local O=${C5_O:-$H/perf/c5a-b2} L
  L=$O/c5a-spec-b2.log
  local C5=caplifive_vp2c_1fc2cd5aa C5SHA=5b7b1edcd9bcd2521bd4bd59107747e268487056f1bfcd96403fd612eeb6d764
  local B3=caplifive_vp2c_e7af48457 B3SHA=d6ed97b6d341ea058d497838e43558900823c36591c44046d3cfc0b7f55b423b
  local T=$H/wt-spec/capstone/tests/rtl-smoke SB=$H/wt-spec/capstone/ports/spec-cpu2017
  local BOOK=${C5_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} FLASH=${C5_FLASH:-python3 $T/drivers/flash-bitstream.py}
  local RB=${C5_READBACK:-python3 $SB/results/2026-10-28-board-test/b4/nv-readback.py}
  local STORE=${C5_STOREHAS:-python3 $T/vp2-kit-2026-10-17/board-tier12/batch/store-has.py}
  local BUSY=${C5_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2}
  local CHAIN=${C5_CHAIN:-bash $SB/board/chain-board.sh} KIT=$H/t3-smoke/bkit17-perf-c5a-b2 SPEC=${C5_SPECOUT:-$H/spec/board/out-c5a-b2}
  local PID=${C5S_PID:?c5a-b2.sh pid} spec=x t=0
  export FPGA_BOARD=2 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url-board2
  log() { echo "CS $(date '+%F %T') $*" >> "$L"; }
  log "START pid $$, waiting for c5a-b2.sh pid $PID"
  [ -n "${C5_PREREG:-}" ] && git -C "$H/wt-spec" branch -r --contains "$C5_PREREG" 2>/dev/null | grep -q origin/spec-silicon || { log "STOP: prereg"; return 2; }
  while kill -0 "$PID" 2>/dev/null; do [ $t -ge 86400 ] && { log "STOP: c5a-b2.sh still running"; return 2; }; sleep 60; t=$((t+60)); done
  local KL=${C5S_KL:-$O/c5a-b2.log}
  sed -n '/START pid 3690279/,$p' "$KL" | grep -q "R67 GRADE rc=0" || { log "STOP: R-67 not FIXED in the kit session"; return 2; }
  sed -n '/START pid 3690279/,$p' "$KL" | grep -q "HARD STOP" && { log "STOP: a HARD STOP in the kit session"; return 2; }
  sed -n '/START pid 3690279/,$p' "$KL" | grep -q "^C5 .* DONE" || { log "STOP: the kit session did not reach DONE"; return 2; }
  sed -n '/START pid 3690279/,$p' "$KL" | grep -q "SPEC on C.5a BEGIN" && { log "STOP: the kit session ran SPEC itself"; return 2; }
  sed -n '/START pid 3690279/,$p' "$KL" | grep -q "OBJB GRADE rc=0" || { log "STOP: the ObjBounds kit did not grade rc 0"; return 2; }
  grep -q "WRONG BITSTREAM\|NO DATA" "${C5S_HPM:-$O/grade-hpm.txt}" && { log "STOP: hpm grade reads WRONG BITSTREAM or NO DATA"; return 2; }
  [ "$(sha256sum "$H/bitstreams/$C5.bit" | cut -d' ' -f1)" = $C5SHA ] || { log "STOP: C.5a .bit hash"; return 2; }
  flash() {
    local n=$1 s=$2 b r
    $BUSY > /dev/null 2>&1 || { log "flash $n: board busy"; return 1; }
    $STORE no-such-bitstream-$$.bit >> "$L" 2>&1; r=$?; [ $r = 1 ] || { log "flash $n: store-has control rc=$r"; return 1; }
    b=$($BOOK book 2 30 "vp2 $n REFLASH (board 2)") || b=
    log "FLASH BEGIN $n booking ${b:-none}"; $FLASH "$n.bit" "$H/bitstreams/$n.bit" "$s" >> "$L" 2>&1; r=$?
    [ -n "$b" ] && $BOOK unbook "$b" >> "$L" 2>&1; log "FLASH END $n rc=$r"; [ $r = 0 ] || return 1
    $RB "$n.bit" "$s" >> "$L" 2>&1; r=$?; case $r in 0|3) log "READBACK $n ok rc=$r";; *) log "READBACK $n rc=$r"; return 1;; esac
  }
  if flash $C5 $C5SHA; then
    log "SPEC on C.5a BEGIN"
    SB_KIT=$KIT SB_BITSTREAM=1fc2cd5aa $CHAIN 2 "$O/c5a-spec-b1.list" "$SPEC" >> "$L" 2>&1; spec=$?
    log "SPEC on C.5a END chain rc=$spec"
    CAPSTONE_WORK=$H C5_OUT=$SPEC python3 "$O/c5a-spec-grade.py" > "$O/grade-spec.txt" 2>&1; log "SPEC GRADE rc=$?"; sed 's/^/  /' "$O/grade-spec.txt" >> "$L"
    # The hpm kit again on C.5a, a fresh flash and session (Addendum 6): is the extra L1D miss the bitstream or the
    # first session's board state? Graded by 5b6fb6a4ec9b's hpm-grade.py against run-b1-bs3.
    local HK=$H/perf/hpm/kit-289ad4e5/images RUNNER=/tmp/capstone/wt-b2/capstone/tests/rtl-smoke/sup-bare-2026-10-03-ladder/run_sup_bare.py i r
    local bid2; bid2=$($BOOK book 2 30 "vp2 hpm kit repeat on C.5a (board 2)") || bid2=
    mkdir -p "$O/hpm-r2"
    for i in hpm-5 hpm-3 hpm-4 hpm-6 hpm-1 hpm-2 hpm-7; do
      ( export FPGA_URL; FPGA_URL=$(cat "$CAPSTONE_FPGA_URL_FILE")
        FPGA_BITSTREAM=$C5.bit SUP_IMG=$HK/$i.bin SUP_REC_ADDR=0x$(sed "s/^0x//" "$HK/$i.rec") SUP_OUT=$O/hpm-r2/$i \
          SUP_TIMEOUT=90 timeout 900 ${C5_RUN:-python3 $RUNNER} > "$O/hpm-r2/$i.stderr" 2>&1 ); r=$?
      echo $r > "$O/hpm-r2/$i.rc"; log "HPM R2 $i rc=$r"
    done
    [ -n "$bid2" ] && $BOOK unbook "$bid2" >> "$L" 2>&1
    python3 "$H/perf/c5a-kits-5b6fb6a4/grade/hpm-grade.py" "$O/hpm-r2" c5a "$H/perf/hpm/run-b1-bs3" > "$O/grade-hpm-r2.txt" 2>&1; log "HPM R2 GRADE rc=$?"; sed 's/^/  /' "$O/grade-hpm-r2.txt" >> "$L"
  else log "C.5a flash failed"; fi
  flash $B3 $B3SHA || { log "STOP: the flash back to bitstream 3 failed"; return 1; }
  log "DONE spec $spec; board 2 back on bitstream 3"
}
main "$@"; exit $?
