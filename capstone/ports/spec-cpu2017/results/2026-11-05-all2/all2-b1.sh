#!/bin/bash
# all2-b1.sh: the all-together run plus the five outside-time switches, on board 1 (results/2026-11-05-all2/PREREG.md,
# A2_PREREG). Kit bkit23-all2-b1 = bkit20-all-b1 (devmk images on C.5a, fault-ahead 64) with module 2f81e8a5842010b3
# (wt-fa perf/retire-sweep-vp2 2188cab26d40; params retire_exit_sweep=3 image_rootfs=1 step_budget_ns=80000000), the
# exact-grant launcher 0bde5fd8e0c9c374, and firmware quantum 8,000,000. Reference: out-all-b1 (same board, bitstream,
# images).
#  1. Wait for all-b1.sh (A2_WAITPID) to end with DONE in its own run (board 1 back on bitstream 3).
#  2. Check the QEMU gate of module + launcher (modrs/qeg-*, six stages).
#  3. Flash C.5a (sealed sha256), read it back.
#  4. gcc alone (the canary: rootfs + exact grant never ran together on silicon); grade it; stop if Y0 fails.
#  5. The other five stages; grade.
#  6. Flash bitstream 3 back.
# Stubs: A2_BUSY A2_BOOK A2_FLASH A2_READBACK A2_STOREHAS A2_CHAIN A2_O A2_OUT A2_WAITLOG A2_POLL_S A2_QGATE A2_GRADE.
main() {
  set -u
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}
  local O=${A2_O:-$H/perf/all2-b1} L
  L=$O/all2-b1.log
  local C5=caplifive_vp2c_1fc2cd5aa C5SHA=5b7b1edcd9bcd2521bd4bd59107747e268487056f1bfcd96403fd612eeb6d764
  local B3=caplifive_vp2c_e7af48457 B3SHA=d6ed97b6d341ea058d497838e43558900823c36591c44046d3cfc0b7f55b423b
  local T=$H/wt-spec/capstone/tests/rtl-smoke SB=$H/wt-spec/capstone/ports/spec-cpu2017 MOD=2f81e8a5842010b3 EGV=0bde5fd8e0c9c374
  local BOOK=${A2_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} FLASH=${A2_FLASH:-python3 $T/drivers/flash-bitstream.py}
  local RB=${A2_READBACK:-python3 $SB/results/2026-10-28-board-test/b4/nv-readback.py}
  local STORE=${A2_STOREHAS:-python3 $T/vp2-kit-2026-10-17/board-tier12/batch/store-has.py}
  local BUSY=${A2_BUSY:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1}
  local CHAIN=${A2_CHAIN:-bash $H/perf/xp-b2/board/chain-board.sh} KIT=$H/t3-smoke/bkit23-all2-b1
  local OUT=${A2_OUT:-$H/spec/board/out-all2-b1} GRADE=${A2_GRADE:-python3 $O/all2-grade.py} spec=skipped t r
  export FPGA_BOARD=1 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url
  log() { echo "A2 $(date '+%F %T') $*" >> "$L"; }
  mkdir -p "$O"; log "START pid $$"
  [ "$(sha256sum "$H/bitstreams/$C5.bit" | cut -d' ' -f1)" = $C5SHA ] || { log "STOP: C.5a .bit hash"; return 2; }
  [ "$(sha256sum "$H/bitstreams/$B3.bit" | cut -d' ' -f1)" = $B3SHA ] || { log "STOP: bitstream 3 .bit hash"; return 2; }
  ( cd "$KIT" && sha256sum -c --quiet SHA256SUMS ) && grep -q "^BOARD_BITSTREAM=$C5$" "$KIT/board.env" \
    && [ "$(sha256sum "$KIT/adapter/module/capstone_vm.ko" | cut -c1-16)" = $MOD ] \
    && [ "$(sha256sum "$KIT/adapter/capstone-vexec" | cut -c1-16)" = $EGV ] \
    && grep -q "CAPSTONE_BOOT_QUANTUM=8000000 -DCAPSTONE_SUPERVISOR_QUANTUM=8000000" "$KIT/board.env" \
    && [ "$(grep -v '^#' "$KIT/adapter/module/params" | sort | tr '\n' ' ')" = "fault_ahead_max=64 image_rootfs=1 retire_exit_sweep=3 step_budget_ns=80000000 " ] \
    || { log "STOP: kit bkit23-all2-b1"; return 2; }
  [ -f "$O/all2-grade.py" ] && [ -s "$O/gcc.list" ] && [ -s "$O/rest.list" ] || { log "STOP: grader or lists"; return 2; }
  [ -e "$OUT" ] && { log "STOP: $OUT exists"; return 2; }
  [ -n "${A2_PREREG:-}" ] && git -C "$H/wt-spec" branch -r --contains "$A2_PREREG" 2>/dev/null | grep -q origin/spec-silicon \
    || { log "STOP: pre-registration ${A2_PREREG:-unset} is not on origin/spec-silicon"; return 2; }
  log "preconditions ok; waiting for all-b1.sh (pid ${A2_WAITPID:-none})"
  t=0; while [ -n "${A2_WAITPID:-}" ] && kill -0 "$A2_WAITPID" 2>/dev/null; do
    [ $t -ge 129600 ] && { log "STOP: all-b1.sh still runs"; return 2; }; sleep "${A2_POLL_S:-60}"; t=$((t + ${A2_POLL_S:-60})); done
  sed -n "/START pid ${A2_WAITPID:-none}\$/,\$p" "${A2_WAITLOG:-$H/perf/all-b1/all-b1.log}" | grep -q "^AL .* DONE" \
    || { log "STOP: all-b1.sh (pid ${A2_WAITPID:-none}) did not end with DONE"; return 2; }
  python3 -c 'import json,sys
for b in sys.argv[2:]:
    r=json.load(open(sys.argv[1]+"/qeg-"+b+"/result.json")); a=r["arms"]["capstone-spatial"]
    assert r.get("done") and all(c["rc"]==0 for c in a["commands"]) and all(v["verdict"]=="PASS" for v in a["validation"]), b' \
    "${A2_QGATE:-$H/modrs}" gcc imagick nab lbm xz mcf >> "$L" 2>&1 || { log "STOP: the QEMU gate (modrs/qeg-*) has not passed"; return 2; }
  log "QEMU gate passed"
  $BUSY > /dev/null 2>&1 || { log "STOP: board 1 busy"; return 2; }
  flash() {
    local n=$1 s=$2 b r
    $STORE no-such-bitstream-$$.bit >> "$L" 2>&1; r=$?; [ $r = 1 ] || { log "flash $n: store-has control rc=$r"; return 1; }
    b=$($BOOK book 1 30 "vp2 $n REFLASH (board 1)") || b=
    log "FLASH BEGIN $n booking ${b:-none}"; $FLASH "$n.bit" "$H/bitstreams/$n.bit" "$s" >> "$L" 2>&1; r=$?
    [ -n "$b" ] && $BOOK unbook "$b" >> "$L" 2>&1; log "FLASH END $n rc=$r"; [ $r = 0 ] || return 1
    $RB "$n.bit" "$s" >> "$L" 2>&1; r=$?; case $r in 0|3) log "READBACK $n ok rc=$r";; *) log "READBACK $n rc=$r"; return 1;; esac
  }
  if flash $C5 $C5SHA; then
    log "CHAIN gcc BEGIN"
    SB_KIT=$KIT SB_BITSTREAM=1fc2cd5aa SB_MODULE=$MOD SB_VEXEC=$EGV $CHAIN 1 "$O/gcc.list" "$OUT" >> "$L" 2>&1; r=$?
    log "CHAIN gcc END rc=$r"
    CAPSTONE_WORK=$H A2_ROOT="$OUT" $GRADE --canary > "$O/grade-gcc.txt" 2>&1; r=$?; log "GRADE gcc rc=$r"; sed 's/^/  /' "$O/grade-gcc.txt" >> "$L"
    if [ $r = 0 ] && [ ! -e "$OUT/.chain-b1-stopped" ]; then
      log "CHAIN rest BEGIN"
      SB_KIT=$KIT SB_BITSTREAM=1fc2cd5aa SB_MODULE=$MOD SB_VEXEC=$EGV $CHAIN 1 "$O/rest.list" "$OUT" >> "$L" 2>&1; spec=$?
      log "CHAIN rest END rc=$spec"
    else log "STOP after gcc: the canary did not pass (grade rc=$r)"; spec=canary; fi
    CAPSTONE_WORK=$H A2_ROOT="$OUT" $GRADE > "$O/grade.txt" 2>&1; log "GRADE rc=$?"; sed 's/^/  /' "$O/grade.txt" >> "$L"
  else log "C.5a flash failed"; fi
  flash $B3 $B3SHA || { log "STOP: the flash back to bitstream 3 failed; board 1 left as is"; return 1; }
  log "DONE spec $spec; board 1 back on bitstream 3"
}
main "$@"; exit $?
