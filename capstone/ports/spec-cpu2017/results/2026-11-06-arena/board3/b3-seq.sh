#!/bin/bash
# b3-seq.sh: board 3's first queue (arena Addendum 10), one task:
#  1. bitstream 3 (the known-good fallback) flashed non-volatile and read back, the board's first flash;
#  2. gcc train c3f: perf/b3/gt-run-ga2.sh 3 (flashes C.5a, runs, flashes bitstream 3 back);
#  3. the THP mcf runs: perf/b3/gt-run-thp.sh 3.
# Stubs: B3_BOOK B3_FLASH B3_READBACK, and the GT_* stubs pass through to the drivers.
main() {
  set -u
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone} HERE; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); local O=${B3_O:-$H/perf/b3} L rc
  local B3N=caplifive_vp2c_e7af48457 B3SHA=d6ed97b6d341ea058d497838e43558900823c36591c44046d3cfc0b7f55b423b
  local T=$H/wt-spec/capstone/tests/rtl-smoke SB=$H/wt-spec/capstone/ports/spec-cpu2017
  local BOOK=${B3_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} FLASH=${B3_FLASH:-python3 $T/drivers/flash-bitstream.py}
  local RB=${B3_READBACK:-python3 $SB/results/2026-10-28-board-test/b4/nv-readback.py} UF=$HOME/.claude-kisp/secrets/fpga-console-url-board3 b
  L=$O/b3-seq.log; log() { echo "B3 $(date '+%F %T') $*" >> "$L"; }; log "START pid $$"
  [ -n "${B3_PREREG:-}" ] && git -C "$H/wt-spec" branch -r --contains "$B3_PREREG" 2>/dev/null | grep -q origin/spec-silicon \
    || { log "STOP: pre-registration ${B3_PREREG:-unset} is not on origin/spec-silicon"; return 2; }
  [ -s "$UF" ] || { log "STOP: no board-3 console URL file"; return 2; }
  [ "$(sha256sum "$H/bitstreams/$B3N.bit" | cut -d' ' -f1)" = $B3SHA ] || { log "STOP: bitstream 3 .bit hash"; return 2; }
  export FPGA_BOARD=3 CAPSTONE_FPGA_URL_FILE=$UF BOARD_URL_FILE=$UF
  b=$($BOOK book 3 30 "vp2 $B3N first flash (board 3, fallback)") || b=
  log "FLASH BEGIN $B3N booking ${b:-none}"; $FLASH "$B3N.bit" "$H/bitstreams/$B3N.bit" $B3SHA >> "$L" 2>&1; rc=$?
  [ $rc = 0 ] || { log "FLASH rc=$rc, retrying once in 60 s"; sleep "${FLASH_RETRY_S:-60}"; $FLASH "$B3N.bit" "$H/bitstreams/$B3N.bit" $B3SHA >> "$L" 2>&1; rc=$?; }
  [ -n "$b" ] && $BOOK unbook "$b" >> "$L" 2>&1; log "FLASH END rc=$rc"; [ $rc = 0 ] || { log "STOP: first flash failed"; return 1; }
  $RB "$B3N.bit" $B3SHA >> "$L" 2>&1; rc=$?; case $rc in 0|3) log "READBACK $B3N ok rc=$rc";; *) log "STOP: READBACK rc=$rc"; return 1;; esac
  log "GA2 BEGIN"; env GT_PREREG=$B3_PREREG GT_OUT=${B3_GA2_OUT:-$H/spec/board/out-ga2-b3} GT_O=${B3_GA2_O:-$O/ga2} \
    ${B3_GA2:-bash $HERE/gt-run-ga2.sh} 3; rc=$?; log "GA2 END rc=$rc"
  log "THP BEGIN"; env GT_PREREG=$B3_PREREG GT_OUT=${B3_THP_OUT:-$H/spec/board/out-thp-b3} GT_O=${B3_THP_O:-$O/thp} \
    ${B3_THP:-bash $HERE/gt-run-thp.sh} 3; rc=$?; log "THP END rc=$rc"
  log "DONE"
}
main "$@"; exit $?
