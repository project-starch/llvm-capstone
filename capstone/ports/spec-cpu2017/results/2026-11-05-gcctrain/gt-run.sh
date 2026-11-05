#!/bin/bash
# gt-run.sh <board 1|2>: gcc train (train01) on the counter kit bkit26-gt, results/2026-11-05-gcctrain/PREREG.md (GT_PREREG).
# Kit bkit26-gt = bkit23-all2-b1 (the all2 configuration) with module b5ea7d846154fa47 (perf/exit-timing-vp2 3fbed8df1fb8,
# -DCAPSTONE_VM_BENCH_HPM: hpm and exit-timing lines per context, hpm_snapshot) and the counter firmware (wt-wrap-hpm,
# -DCAPSTONE_HPM_EVENTS).
#  1. Wait for the board's previous session (GT_WAITPID) to end with DONE in its own run (GT_WAITLOG, prefix GT_WAITPFX).
#  2. Flash C.5a (sealed sha256), read it back.
#  3. The board's list (b<board>.list: board 2 Temporal, board 1 Plain), then flash bitstream 3 back.
# Stubs: GT_BUSY GT_BOOK GT_FLASH GT_READBACK GT_STOREHAS GT_CHAIN GT_O GT_OUT GT_POLL_S.
main() {
  set -u
  local BRD=${1:?board} H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}
  local O=${GT_O:-$H/perf/gt} L
  L=$O/gt-b$BRD.log
  local C5=caplifive_vp2c_1fc2cd5aa C5SHA=5b7b1edcd9bcd2521bd4bd59107747e268487056f1bfcd96403fd612eeb6d764
  local B3=caplifive_vp2c_e7af48457 B3SHA=d6ed97b6d341ea058d497838e43558900823c36591c44046d3cfc0b7f55b423b
  local T=$H/wt-spec/capstone/tests/rtl-smoke SB=$H/wt-spec/capstone/ports/spec-cpu2017 MOD=65ede944a87ea742 EGV=0bde5fd8e0c9c374
  local BOOK=${GT_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} FLASH=${GT_FLASH:-python3 $T/drivers/flash-bitstream.py}
  local RB=${GT_READBACK:-python3 $SB/results/2026-10-28-board-test/b4/nv-readback.py}
  local STORE=${GT_STOREHAS:-python3 $T/vp2-kit-2026-10-17/board-tier12/batch/store-has.py}
  local BUSY
  case $BRD in
    1) BUSY=${GT_BUSY:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1};;
    2) BUSY=${GT_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2};;
    *) echo "board $BRD"; return 2;;
  esac
  local CHAIN=${GT_CHAIN:-bash $H/perf/xp-b2/board/chain-board.sh} KIT=$H/t3-smoke/bkit26-gt
  local OUT=${GT_OUT:-$H/spec/board/out-gt-b$BRD}
  local spec=skipped t r LIST
  LIST=$O/b$BRD.list
  # each board has its own console URL file (board 2: -board2); the board 1 file refuses board 2 (fpga_console.py)
  local UF=$HOME/.claude-kisp/secrets/fpga-console-url; [ $BRD = 2 ] && UF=$UF-board2
  export FPGA_BOARD=$BRD CAPSTONE_FPGA_URL_FILE=$UF
  [ -s "$CAPSTONE_FPGA_URL_FILE" ] || { echo "no console URL file for board $BRD"; return 2; }
  log() { echo "GT $(date '+%F %T') $*" >> "$L"; }
  mkdir -p "$O"; log "START pid $$ board $BRD"
  [ "$(sha256sum "$H/bitstreams/$C5.bit" | cut -d' ' -f1)" = $C5SHA ] || { log "STOP: C.5a .bit hash"; return 2; }
  [ "$(sha256sum "$H/bitstreams/$B3.bit" | cut -d' ' -f1)" = $B3SHA ] || { log "STOP: bitstream 3 .bit hash"; return 2; }
  ( cd "$KIT" && sha256sum -c --quiet SHA256SUMS ) && grep -q "^BOARD_BITSTREAM=$C5$" "$KIT/board.env" \
    && [ "$(sha256sum "$KIT/adapter/module/capstone_vm.ko" | cut -c1-16)" = $MOD ] \
    && [ "$(sha256sum "$KIT/adapter/capstone-vexec" | cut -c1-16)" = $EGV ] \
    && grep -q "CAPSTONE_BOOT_QUANTUM=8000000 -DCAPSTONE_SUPERVISOR_QUANTUM=8000000 -DCAPSTONE_HPM_EVENTS" "$KIT/board.env" && grep -q "^BOARD_FW_WRAP=$H/wt-wrap-hpm$" "$KIT/board.env" \
    && [ "$(grep -v '^#' "$KIT/adapter/module/params" | sort | tr '\n' ' ')" = "fault_ahead_max=64 image_rootfs=1 retire_exit_sweep=3 step_budget_ns=80000000 " ] \
    || { log "STOP: kit bkit26-gt"; return 2; }
  [ -s "$LIST" ] || { log "STOP: list"; return 2; }
  [ -e "$OUT" ] && { log "STOP: $OUT exists"; return 2; }
  [ -n "${GT_PREREG:-}" ] && git -C "$H/wt-spec" branch -r --contains "$GT_PREREG" 2>/dev/null | grep -q origin/spec-silicon \
    || { log "STOP: pre-registration ${GT_PREREG:-unset} is not on origin/spec-silicon"; return 2; }
  log "preconditions ok; waiting for pid ${GT_WAITPID:-none}"
  t=0; while [ -n "${GT_WAITPID:-}" ] && kill -0 "$GT_WAITPID" 2>/dev/null; do
    [ $t -ge 172800 ] && { log "STOP: pid $GT_WAITPID still runs"; return 2; }; sleep "${GT_POLL_S:-60}"; t=$((t + ${GT_POLL_S:-60})); done
  # only when there was a session to wait for (Addendum 1: the runs start at once, GT_WAITPID unset)
  [ -z "${GT_WAITPID:-}" ] || sed -n "/START pid ${GT_WAITPID:-none}\$/,\$p;/START pid ${GT_WAITPID:-none} /,\$p" "${GT_WAITLOG:-/dev/null}" | grep -q "^${GT_WAITPFX:-XX} .* DONE" \
    || { log "STOP: pid ${GT_WAITPID:-none} did not end with DONE"; return 2; }
  $BUSY > /dev/null 2>&1 || { log "STOP: board $BRD busy"; return 2; }
  flash() {
    local n=$1 s=$2 b r
    $STORE no-such-bitstream-$$.bit >> "$L" 2>&1; r=$?; [ $r = 1 ] || { log "flash $n: store-has control rc=$r"; return 1; }
    b=$($BOOK book $BRD 30 "vp2 $n REFLASH (board $BRD)") || b=
    log "FLASH BEGIN $n booking ${b:-none}"; $FLASH "$n.bit" "$H/bitstreams/$n.bit" "$s" >> "$L" 2>&1; r=$?
    [ -n "$b" ] && $BOOK unbook "$b" >> "$L" 2>&1; log "FLASH END $n rc=$r"; [ $r = 0 ] || return 1
    $RB "$n.bit" "$s" >> "$L" 2>&1; r=$?; case $r in 0|3) log "READBACK $n ok rc=$r";; *) log "READBACK $n rc=$r"; return 1;; esac
  }
  if flash $C5 $C5SHA; then
    r=0
    if [ $r = 0 ] && [ ! -e "$OUT/.chain-b$BRD-stopped" ]; then
      log "CHAIN rest BEGIN"
      SB_KIT=$KIT SB_BITSTREAM=1fc2cd5aa SB_MODULE=$MOD SB_VEXEC=$EGV $CHAIN $BRD "$LIST" "$OUT" >> "$L" 2>&1; spec=$?
      log "CHAIN rest END rc=$spec"
    fi
  else log "C.5a flash failed"; fi
  flash $B3 $B3SHA || { log "STOP: the flash back to bitstream 3 failed; board $BRD left as is"; return 1; }
  log "DONE spec $spec; board $BRD back on bitstream 3"
}
main "$@"; exit $?
