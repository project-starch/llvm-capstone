#!/bin/bash
# xp-b2.sh: the outside-time levers on board 2, bitstream 3 (results/2026-11-05-outside-levers/PREREG.md, XP_PREREG),
# after board 2's C.5a session (c5a-spec-b2.sh, XP_WAITPID) has ended with DONE and board 2 is back on bitstream 3.
# Module 2f81e8a5842010b3 (wt-fa perf/retire-sweep-vp2 2188cab26d40) in every kit bkit22-<kit>-b2; one chain per kit:
#   gcc ctl, gcc sweep, gcc nosweep, gcc root, imagick root, nab q8, lbm allmod. No flash: board 2 stays on bitstream 3.
# Then the grader. dt-b2.sh (stage A) is relaunched to wait on this script's pid.
# Stubs for the dry run: XP_BUSY XP_RB XP_CHAIN XP_O XP_OUTROOT XP_WAITLOG XP_POLL_S.
main() {
  set -u
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}
  local O=${XP_O:-$H/perf/xp-b2} L
  L=$O/xp-b2.log
  local B3=caplifive_vp2c_e7af48457 B3SHA=d6ed97b6d341ea058d497838e43558900823c36591c44046d3cfc0b7f55b423b MOD=2f81e8a5842010b3
  local SB=$H/wt-spec/capstone/ports/spec-cpu2017 T=$H/t3-smoke
  local RB=${XP_RB:-python3 $SB/results/2026-10-28-board-test/b4/nv-readback.py}
  local BUSY=${XP_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2}
  local CHAIN=${XP_CHAIN:-bash $H/perf/fa/board/chain-board.sh} OR=${XP_OUTROOT:-$H/spec/board} k li t r ran=0
  export FPGA_BOARD=2 CAPSTONE_FPGA_URL_FILE=$HOME/.claude-kisp/secrets/fpga-console-url-board2
  log() { echo "XP $(date '+%F %T') $*" >> "$L"; }
  mkdir -p "$O"; log "START pid $$"
  for k in ctl sweep nosweep root q8 allmod; do
    ( cd "$T/bkit22-$k-b2" && sha256sum -c --quiet SHA256SUMS ) && [ "$(sha256sum "$T/bkit22-$k-b2/adapter/module/capstone_vm.ko" | cut -c1-16)" = $MOD ] \
      && grep -q "^BOARD_BITSTREAM=$B3$" "$T/bkit22-$k-b2/board.env" || { log "STOP: kit bkit22-$k-b2"; return 2; }
  done
  grep -q "CAPSTONE_BOOT_QUANTUM=8000000 -DCAPSTONE_SUPERVISOR_QUANTUM=8000000" "$T/bkit22-q8-b2/board.env" || { log "STOP: q8 quantum"; return 2; }
  for li in gcc imagick nab lbm; do [ -s "$O/$li.list" ] || { log "STOP: list $li"; return 2; }; done
  [ -f "$O/xp-grade.py" ] || { log "STOP: grader"; return 2; }
  for k in ctl sweep nosweep root q8 allmod; do [ -e "$OR/out-xp-$k" ] && { log "STOP: $OR/out-xp-$k exists"; return 2; }; done
  [ -n "${XP_PREREG:-}" ] && git -C "$H/wt-spec" branch -r --contains "$XP_PREREG" 2>/dev/null | grep -q origin/spec-silicon \
    || { log "STOP: pre-registration ${XP_PREREG:-unset} is not on origin/spec-silicon"; return 2; }
  log "preconditions ok; waiting for the C.5a session (pid ${XP_WAITPID:-none})"
  t=0; while [ -n "${XP_WAITPID:-}" ] && kill -0 "$XP_WAITPID" 2>/dev/null; do
    [ $t -ge 86400 ] && { log "STOP: the C.5a session still runs"; return 2; }; sleep "${XP_POLL_S:-60}"; t=$((t + ${XP_POLL_S:-60})); done
  grep -q "^CS .* DONE" "${XP_WAITLOG:-$H/perf/c5a-b2/c5a-spec-b2.log}" || { log "STOP: the C.5a session did not end with DONE"; return 2; }
  # The QEMU gate of this module (PREREG): gcc, imagick, nab and lbm finished with rc 0 and specdiff PASS.
  python3 -c 'import json,sys
for b in sys.argv[2:]:
    r=json.load(open(sys.argv[1]+"/q3-"+b+"/result.json")); a=r["arms"]["capstone-spatial"]
    assert r.get("done") and all(c["rc"]==0 for c in a["commands"]) and all(v["verdict"]=="PASS" for v in a["validation"]), b' \
    "${XP_QGATE:-$H/modrs}" gcc imagick nab lbm >> "$L" 2>&1 || { log "STOP: the QEMU gate (modrs/q3-*) has not passed"; return 2; }
  log "QEMU gate passed"
  $BUSY > /dev/null 2>&1 || { log "STOP: board 2 busy"; return 2; }
  $RB "$B3.bit" "$B3SHA" >> "$L" 2>&1; r=$?; case $r in 0|3) log "READBACK $B3 ok rc=$r";; *) log "STOP: board 2 is not on bitstream 3 (readback rc=$r)"; return 2;; esac
  for k_li in ctl:gcc sweep:gcc nosweep:gcc root:gcc root:imagick q8:nab allmod:lbm; do
    k=${k_li%%:*}; li=${k_li##*:}
    log "CHAIN $k $li BEGIN"
    SB_KIT=$T/bkit22-$k-b2 SB_BITSTREAM=e7af48457 SB_MODULE=$MOD $CHAIN 2 "$O/$li.list" "$OR/out-xp-$k" >> "$L" 2>&1; r=$?
    log "CHAIN $k $li END rc=$r"; ran=$((ran + 1))
    [ -e "$OR/out-xp-$k/.chain-b2-stopped" ] && { log "STOP: chain $k $li stopped (board busy)"; break; }
  done
  CAPSTONE_WORK=$H XP_ROOT=$OR python3 "$O/xp-grade.py" > "$O/grade.txt" 2>&1; log "GRADE rc=$?"; sed 's/^/  /' "$O/grade.txt" >> "$L"
  log "DONE chains $ran; board 2 on bitstream 3"
}
main "$@"; exit $?
