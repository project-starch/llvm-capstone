#!/bin/bash
# ns-run.sh <board 1|2>: all2 with no exit sweep (retire_exit_sweep=2), results/2026-11-05-nosweep/PREREG.md (NS_PREREG).
# Kit bkit25-nset = bkit23-all2-b1 with the params line retire_exit_sweep=2 and the exit-timing module (Addendum 1).
# The board that runs the probe is NS_PROBE_BOARD (default 2); the other waits for its verdict.
#  1. Wait for the board's previous session (NS_WAITPID) to end with DONE in its own run (NS_WAITLOG, prefix NS_WAITPFX).
#  2. The other board only: wait for the probe verdict (NS_VERDICT, written by the probe board's run): N1 HOLDS, or stop.
#  3. Flash C.5a (sealed sha256), read it back.
#  4. The probe board only: the tag-leak probe stage (probe.list), its verdict to NS_VERDICT; stop unless N1 holds.
#  5. The board's stages (b<board>-rest.list), then flash bitstream 3 back.
# Stubs: NS_BUSY NS_BOOK NS_FLASH NS_READBACK NS_STOREHAS NS_CHAIN NS_O NS_OUT NS_POLL_S NS_VERDICT NS_TLV.
main() {
  set -u
  local BRD=${1:?board} H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}
  local O=${NS_O:-$H/perf/ns} L
  L=$O/ns-b$BRD.log
  local C5=caplifive_vp2c_1fc2cd5aa C5SHA=5b7b1edcd9bcd2521bd4bd59107747e268487056f1bfcd96403fd612eeb6d764
  local B3=caplifive_vp2c_e7af48457 B3SHA=d6ed97b6d341ea058d497838e43558900823c36591c44046d3cfc0b7f55b423b
  local T=$H/wt-spec/capstone/tests/rtl-smoke SB=$H/wt-spec/capstone/ports/spec-cpu2017 MOD=1ae19c5e337fbeb2 EGV=0bde5fd8e0c9c374
  local BOOK=${NS_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} FLASH=${NS_FLASH:-python3 $T/drivers/flash-bitstream.py}
  local RB=${NS_READBACK:-python3 $SB/results/2026-10-28-board-test/b4/nv-readback.py}
  local STORE=${NS_STOREHAS:-python3 $T/vp2-kit-2026-10-17/board-tier12/batch/store-has.py}
  local BUSY
  case $BRD in
    1) BUSY=${NS_BUSY:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1};;
    2) BUSY=${NS_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2};;
    *) echo "board $BRD"; return 2;;
  esac
  local CHAIN=${NS_CHAIN:-bash $H/perf/xp-b2/board/chain-board.sh} KIT=$H/t3-smoke/bkit25-nset
  local OUT=${NS_OUT:-$H/spec/board/out-ns-b$BRD} TLV=${NS_TLV:-python3 $O/tl-verdict.py} VERDICT=${NS_VERDICT:-$O/verdict-N1.txt}
  local spec=skipped t r LIST PB=${NS_PROBE_BOARD:-2}
  LIST=$O/b$BRD-rest.list
  # each board has its own console URL file (board 2: -board2); the board 1 file refuses board 2 (fpga_console.py)
  local UF=$HOME/.claude-kisp/secrets/fpga-console-url; [ $BRD = 2 ] && UF=$UF-board2
  export FPGA_BOARD=$BRD CAPSTONE_FPGA_URL_FILE=$UF
  [ -s "$CAPSTONE_FPGA_URL_FILE" ] || { echo "no console URL file for board $BRD"; return 2; }
  log() { echo "NS $(date '+%F %T') $*" >> "$L"; }
  mkdir -p "$O"; log "START pid $$ board $BRD"
  # the probe board owes the other a verdict on every way out, or the other waits a day for nothing
  [ $BRD = $PB ] && trap '[ -s "$VERDICT" ] || echo "N1 FAILS no probe" > "$VERDICT"' RETURN
  [ "$(sha256sum "$H/bitstreams/$C5.bit" | cut -d' ' -f1)" = $C5SHA ] || { log "STOP: C.5a .bit hash"; return 2; }
  [ "$(sha256sum "$H/bitstreams/$B3.bit" | cut -d' ' -f1)" = $B3SHA ] || { log "STOP: bitstream 3 .bit hash"; return 2; }
  ( cd "$KIT" && sha256sum -c --quiet SHA256SUMS ) && grep -q "^BOARD_BITSTREAM=$C5$" "$KIT/board.env" \
    && [ "$(sha256sum "$KIT/adapter/module/capstone_vm.ko" | cut -c1-16)" = $MOD ] \
    && [ "$(sha256sum "$KIT/adapter/capstone-vexec" | cut -c1-16)" = $EGV ] \
    && grep -q "CAPSTONE_BOOT_QUANTUM=8000000 -DCAPSTONE_SUPERVISOR_QUANTUM=8000000" "$KIT/board.env" \
    && [ "$(grep -v '^#' "$KIT/adapter/module/params" | sort | tr '\n' ' ')" = "fault_ahead_max=64 image_rootfs=1 retire_exit_sweep=2 step_budget_ns=80000000 " ] \
    || { log "STOP: kit bkit25-nset"; return 2; }
  [ -s "$LIST" ] && [ -s "$O/probe.list" ] && [ -f "$O/tl-verdict.py" ] || { log "STOP: lists or verdict script"; return 2; }
  [ -e "$OUT" ] && { log "STOP: $OUT exists"; return 2; }
  [ -n "${NS_PREREG:-}" ] && git -C "$H/wt-spec" branch -r --contains "$NS_PREREG" 2>/dev/null | grep -q origin/spec-silicon \
    || { log "STOP: pre-registration ${NS_PREREG:-unset} is not on origin/spec-silicon"; return 2; }
  log "preconditions ok; waiting for pid ${NS_WAITPID:-none}"
  t=0; while [ -n "${NS_WAITPID:-}" ] && kill -0 "$NS_WAITPID" 2>/dev/null; do
    [ $t -ge 172800 ] && { log "STOP: pid $NS_WAITPID still runs"; return 2; }; sleep "${NS_POLL_S:-60}"; t=$((t + ${NS_POLL_S:-60})); done
  sed -n "/START pid ${NS_WAITPID:-none}\$/,\$p;/START pid ${NS_WAITPID:-none} /,\$p" "${NS_WAITLOG:-/dev/null}" | grep -q "^${NS_WAITPFX:-XX} .* DONE" \
    || { log "STOP: pid ${NS_WAITPID:-none} did not end with DONE"; return 2; }
  if [ $BRD != $PB ]; then
    t=0; until [ -s "$VERDICT" ]; do
      [ $t -ge 86400 ] && { log "STOP: no probe verdict"; return 2; }; sleep "${NS_POLL_S:-60}"; t=$((t + ${NS_POLL_S:-60})); done
    grep -q "^N1 HOLDS$" "$VERDICT" || { log "STOP: probe verdict is not N1 HOLDS"; return 2; }
    log "probe verdict N1 HOLDS"
  fi
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
    if [ $BRD = $PB ]; then
      log "CHAIN probe BEGIN"
      SB_KIT=$KIT SB_BITSTREAM=1fc2cd5aa SB_MODULE=$MOD SB_VEXEC=$EGV $CHAIN $BRD "$O/probe.list" "$OUT" >> "$L" 2>&1
      log "CHAIN probe END rc=$?"
      $TLV "$OUT/999.tagleak-test/guest/board-raw.txt" > "$O/verdict-detail.txt" 2>&1; r=$?
      sed 's/^/  /' "$O/verdict-detail.txt" >> "$L"; log "PROBE verdict rc=$r"
      # the other board waits on this file: only the summary line, written once, after the read
      tail -1 "$O/verdict-detail.txt" > "$VERDICT.tmp"; [ $r = 0 ] || echo "N1 FAILS rc=$r" > "$VERDICT.tmp"; mv "$VERDICT.tmp" "$VERDICT"
    fi
    if [ $r = 0 ] && [ ! -e "$OUT/.chain-b$BRD-stopped" ]; then
      log "CHAIN rest BEGIN"
      SB_KIT=$KIT SB_BITSTREAM=1fc2cd5aa SB_MODULE=$MOD SB_VEXEC=$EGV $CHAIN $BRD "$LIST" "$OUT" >> "$L" 2>&1; spec=$?
      log "CHAIN rest END rc=$spec"
    else log "STOP after the probe (verdict rc=$r)"; spec=probe; fi
  else log "C.5a flash failed"; [ $BRD = $PB ] && echo "N1 FAILS flash" > "$VERDICT"; fi
  flash $B3 $B3SHA || { log "STOP: the flash back to bitstream 3 failed; board $BRD left as is"; return 1; }
  log "DONE spec $spec; board $BRD back on bitstream 3"
}
main "$@"; exit $?
