#!/bin/bash
# all-b1.sh: every optimisation measured so far, together, on board 1 (results/2026-11-04-all/PREREG.md, AL_PREREG):
# the compiler evaluation's devmk images (dev compiler + movc flag) on C.5a (1fc2cd5aa) with the fault-ahead module
# (kit bkit20-all-b1: bkit17-perf-c5a-b1 + module 6e854c46c654c843 + fault_ahead_max=64).
#  1. Wait for board 1's counter boots (hpm-after-fa.sh, AL_WAITPID) to end with "HA END".
#  1b. hpm-8, the L1I counter's positive control, on bitstream 3 (informational).
#  2. Flash C.5a (sealed sha256), read it back.
#  3. SPEC test, all-b1.list, into out-all-b1 (perf/fa/board copies: spec-board.sh with SB_MODULE), graded by all-grade.py.
#  4. Flash bitstream 3 back.
# Stubs: AL_BUSY AL_BOOK AL_FLASH AL_READBACK AL_STOREHAS AL_CHAIN AL_O AL_OUT AL_WAITLOG AL_POLL_S.
main() {
  set -u
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}
  local O=${AL_O:-$H/perf/all-b1} L
  L=$O/all-b1.log
  local C5=caplifive_vp2c_1fc2cd5aa C5SHA=5b7b1edcd9bcd2521bd4bd59107747e268487056f1bfcd96403fd612eeb6d764
  local B3=caplifive_vp2c_e7af48457 B3SHA=d6ed97b6d341ea058d497838e43558900823c36591c44046d3cfc0b7f55b423b
  local T=$H/wt-spec/capstone/tests/rtl-smoke SB=$H/wt-spec/capstone/ports/spec-cpu2017 MOD=6e854c46c654c843
  local BOOK=${AL_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} FLASH=${AL_FLASH:-python3 $T/drivers/flash-bitstream.py}
  local RB=${AL_READBACK:-python3 $SB/results/2026-10-28-board-test/b4/nv-readback.py}
  local STORE=${AL_STOREHAS:-python3 $T/vp2-kit-2026-10-17/board-tier12/batch/store-has.py}
  local BUSY=${AL_BUSY:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1}
  local CHAIN=${AL_CHAIN:-bash $H/perf/fa/board/chain-board.sh} KIT=$H/t3-smoke/bkit20-all-b1
  local OUT=${AL_OUT:-$H/spec/board/out-all-b1} spec=skipped t
  export FPGA_BOARD=1 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url
  log() { echo "AL $(date '+%F %T') $*" >> "$L"; }
  mkdir -p "$O"; log "START pid $$"
  [ "$(sha256sum "$H/bitstreams/$C5.bit" | cut -d' ' -f1)" = $C5SHA ] || { log "STOP: C.5a .bit hash"; return 2; }
  [ "$(sha256sum "$H/bitstreams/$B3.bit" | cut -d' ' -f1)" = $B3SHA ] || { log "STOP: bitstream 3 .bit hash"; return 2; }
  ( cd "$KIT" && sha256sum -c --quiet SHA256SUMS ) && grep -q "^BOARD_BITSTREAM=$C5$" "$KIT/board.env" \
    && [ "$(sha256sum "$KIT/adapter/module/capstone_vm.ko" | cut -c1-16)" = $MOD ] && grep -qx "fault_ahead_max=64" "$KIT/adapter/module/params" \
    || { log "STOP: kit bkit20-all-b1"; return 2; }
  [ -f "$O/all-grade.py" ] && [ -f "$O/all-b1.list" ] || { log "STOP: grader or list"; return 2; }
  [ -n "${AL_PREREG:-}" ] && git -C "$H/wt-spec" branch -r --contains "$AL_PREREG" 2>/dev/null | grep -q origin/spec-silicon \
    || { log "STOP: pre-registration ${AL_PREREG:-unset} is not on origin/spec-silicon"; return 2; }
  log "preconditions ok; waiting for board 1's counter boots (pid ${AL_WAITPID:-none})"
  t=0; while [ -n "${AL_WAITPID:-}" ] && kill -0 "$AL_WAITPID" 2>/dev/null; do
    [ $t -ge 129600 ] && { log "STOP: the counter boots still run"; return 2; }; sleep "${AL_POLL_S:-60}"; t=$((t + ${AL_POLL_S:-60})); done
  grep -q "^HA END" "${AL_WAITLOG:-$H/perf/fa/hpm-after-fa.log}" || { log "STOP: the counter boots did not end"; return 2; }
  $BUSY > /dev/null 2>&1 || { log "STOP: board 1 busy"; return 2; }
  # hpm-8 (land/rtl-c5a-kits c950068d8373): the L1I-miss counter's positive control on bitstream 3, the bitstream of the
  # imagick counter boots just before; informational, gates nothing (hpm PREREG Addendum 4).
  local K8=$H/perf/hpm/kit-c950068d RUNNER=/tmp/capstone/wt-b2/capstone/tests/rtl-smoke/sup-bare-2026-10-03-ladder/run_sup_bare.py b8 r
  [ "$(sha256sum "$RUNNER" | cut -c1-16)" = 102ab84e762cba50 ] || log "hpm-8 skipped: runner hash"
  b8=$($BOOK book 1 20 "vp2 hpm-8 L1I control on bitstream 3 (board 1)") || b8=
  mkdir -p "$O/hpm8"
  ( export FPGA_URL; FPGA_URL=$(cat "$CAPSTONE_FPGA_URL_FILE")
    FPGA_BITSTREAM=$B3.bit SUP_IMG=$K8/images/hpm-8.bin SUP_REC_ADDR=0x$(sed "s/^0x//" "$K8/images/hpm-8.rec") SUP_OUT=$O/hpm8/hpm-8 \
      SUP_TIMEOUT=90 timeout 900 ${AL_RUN:-python3 $RUNNER} > "$O/hpm8/hpm-8.stderr" 2>&1 ); r=$?
  [ -n "$b8" ] && $BOOK unbook "$b8" >> "$L" 2>&1
  python3 "$K8/grade/hpm8-grade.py" "$O/hpm8" > "$O/hpm8.grade.txt" 2>&1; log "HPM8 run rc=$r GRADE rc=$?"; sed 's/^/  /' "$O/hpm8.grade.txt" >> "$L"
  flash() {
    local n=$1 s=$2 b r
    $STORE no-such-bitstream-$$.bit >> "$L" 2>&1; r=$?; [ $r = 1 ] || { log "flash $n: store-has control rc=$r"; return 1; }
    b=$($BOOK book 1 30 "vp2 $n REFLASH (board 1)") || b=
    log "FLASH BEGIN $n booking ${b:-none}"; $FLASH "$n.bit" "$H/bitstreams/$n.bit" "$s" >> "$L" 2>&1; r=$?
    [ -n "$b" ] && $BOOK unbook "$b" >> "$L" 2>&1; log "FLASH END $n rc=$r"; [ $r = 0 ] || return 1
    $RB "$n.bit" "$s" >> "$L" 2>&1; r=$?; case $r in 0|3) log "READBACK $n ok rc=$r";; *) log "READBACK $n rc=$r"; return 1;; esac
  }
  if flash $C5 $C5SHA; then
    log "SPEC all-together BEGIN"
    SB_KIT=$KIT SB_BITSTREAM=1fc2cd5aa SB_MODULE=$MOD $CHAIN 1 "$O/all-b1.list" "$OUT" >> "$L" 2>&1; spec=$?
    log "SPEC all-together END chain rc=$spec"
    CAPSTONE_WORK=$H ALL_OUT=$OUT python3 "$O/all-grade.py" > "$O/grade.txt" 2>&1; log "GRADE rc=$?"; sed 's/^/  /' "$O/grade.txt" >> "$L"
  else log "C.5a flash failed"; fi
  flash $B3 $B3SHA || { log "STOP: the flash back to bitstream 3 failed; board 1 left as is"; return 1; }
  log "DONE spec $spec; board 1 back on bitstream 3"
}
main "$@"; exit $?
