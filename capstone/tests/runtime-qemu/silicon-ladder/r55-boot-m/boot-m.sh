#!/bin/bash
# BOOT M (pre-registered, lane a26c9f01935e): the chain step-6 form (k800, s06agg) on the PRIVATE firmware fw-mtime
# 647dce4c7fce (the pinned 2f49d5b257f7 with only the mtime capability cursor set to SBI_MTIME_ADDR, reported as
# MTMB/MTMC/MTMF). Predicted: MTMB:0x2000000 MTMC:0x2000000 MTMF:0x200bff8; nonzero sched_clock stamp; login; k800 4; s06agg 64.
main() {
  set -u
  local D=/tmp/capstone/accept-fpb/disc/bootM O=/tmp/capstone/s06agg-776 FWM=/tmp/capstone/fpb-diag/fw-mtime.bin RS=$HOME/dev/llvm-capstone/capstone/tests/rtl-smoke
  mkdir -p $D
  [ "$(sha256sum $FWM | cut -c1-12)" = 647dce4c7fce ] || { echo "fw mismatch"; return 1; }
  cd $RS || return 1
  export FPGA_BITSTREAM=caplifive_fpb_afcd19bea.bit
  export FPGA_URL="$(cat ~/.claude-kisp/secrets/fpga-console-url)" FPGA_FW=$FWM PREFLIGHT=0
  export SQLITE_STAGE_OVERLAY=$O/ref SQLITE_STAGE_TIMEOUT=300 SQLITE_IDLE_S=300 ENTRY_STALL_S=420 EARLY_HALT_CONTROL=0 WEDGE_TRACER=0 HALT_MUX_READS=0
  export K800_HASH=589ceee3853c6092 K800_ORACLES=$HOME/capstone-artifacts/k800-relinked-0x20000/orc PREFLIGHT_ORACLES=$HOME/capstone-artifacts/k800-relinked-0x20000/orc
  export SQLITE_STAGE_DOMS="/test-domains/lpc|k800:/test-domains/k800.dom,/test-domains/lpc|s06agg:/test-domains/s06agg.dom"
  export PROBE_SCOPED_OUT=$D/boot.txt PROBE_RAW_OUT=$D/boot-raw.txt
  timeout 1800 python3 -m fpga_driver.run_sqlite_stages_fpga > $D/driver.log 2>&1 &
  local RUNNER=$!
  ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420 bash board-watchdog.sh "$D/driver.log" 600 "$RUNNER" > $D/watchdog.log 2>&1 &
  local WD=$!
  wait $RUNNER; local r=$?; kill $WD 2>/dev/null; wait $WD 2>/dev/null
  echo "BOOT_M rc=$r $(date +%H:%M:%S)"
  return $r
}
main "$@"; exit $?
