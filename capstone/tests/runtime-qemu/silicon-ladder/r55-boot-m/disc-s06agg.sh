#!/bin/bash
# DISCRIMINATOR after a1's Linux hang on the FP respin (2026-10-15): the chain's step 6 unchanged -- k800 then s06agg on
# the PINNED firmware 2f49d5b257f7, which booted Linux and read k800 4 / s06agg 64 on 0568f93a9 (2026-10-11 17:42).
# Same firmware bytes, only the bitstream differs. Prediction, written before the boot:
#   H-bitstream: the same hang -- '[ cut here ]' right after the sched_clock line, no login -- on every attempt.
#   H-firmware (today's rebuilt a1 firmware is at fault): Linux reaches login, k800 4, s06agg 64.
main() {
  set -u
  local D=/tmp/capstone/accept-fpb/disc/s06agg O=/tmp/capstone/s06agg-776 RS=$HOME/dev/llvm-capstone/capstone/tests/rtl-smoke
  mkdir -p $D
  [ "$(sha256sum $O/fw_payload.bin | cut -c1-12)" = 2f49d5b257f7 ] || { echo "fw mismatch"; return 1; }
  cd $RS || return 1
  export FPGA_BITSTREAM=caplifive_fpb_afcd19bea.bit
  export FPGA_URL="$(cat ~/.claude-kisp/secrets/fpga-console-url)" FPGA_FW=$O/fw_payload.bin PREFLIGHT=0
  export SQLITE_STAGE_OVERLAY=$O/ref SQLITE_STAGE_TIMEOUT=300 SQLITE_IDLE_S=300 ENTRY_STALL_S=420 EARLY_HALT_CONTROL=0 WEDGE_TRACER=0 HALT_MUX_READS=0
  export K800_HASH=589ceee3853c6092 K800_ORACLES=$HOME/capstone-artifacts/k800-relinked-0x20000/orc PREFLIGHT_ORACLES=$HOME/capstone-artifacts/k800-relinked-0x20000/orc
  export SQLITE_STAGE_DOMS="/test-domains/lpc|k800:/test-domains/k800.dom,/test-domains/lpc|s06agg:/test-domains/s06agg.dom"
  export PROBE_SCOPED_OUT=$D/boot.txt PROBE_RAW_OUT=$D/boot-raw.txt
  timeout 1800 python3 -m fpga_driver.run_sqlite_stages_fpga > $D/driver.log 2>&1 &
  local RUNNER=$!
  ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420 bash board-watchdog.sh "$D/driver.log" 600 "$RUNNER" > $D/watchdog.log 2>&1 &
  local WD=$!
  wait $RUNNER; local r=$?; kill $WD 2>/dev/null; wait $WD 2>/dev/null
  echo "DISC_S06AGG rc=$r $(date +%H:%M:%S)"
  return $r
}
main "$@"; exit $?
