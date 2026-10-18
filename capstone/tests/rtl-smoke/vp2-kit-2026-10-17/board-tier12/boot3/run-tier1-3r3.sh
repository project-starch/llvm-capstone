#!/bin/bash
# vp2 Tier 1 boot 3-T1 THIRD run (2026-10-18): Tier 1 alone, the repeat with ONE change, the board module 0cc4760d21fde17a
# (Q-26 + Q-30, llvm-capstone 457b78f27e10). Rungs b0-stats, b0-hello and vm-contract; NO vm-suite (Tier 2 is boot 4).
# Images, launcher, monitor 1e9451a and recipe as the repeat; the judge adds the reply-slot branch.
# Usage: run-tier1-3r3.sh <firmware sha256/16 as pre-registered>. T1_* variables are for the stubbed dry run only.
DRV_DIR=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke
DRV_SHA=4ae259caae50b35d                        # sha256/16 of the lane's run_baked_rungs_fpga.py (lane 0b8813639710)
main() {
  set -u
  local T=/tmp/capstone/tier1 D=${T1_OUT:-/tmp/capstone/tier1/run3r3} FW=${T1_FW:-/tmp/capstone/tier1/fw3r3/fw_payload.bin}
  local WANT=${1:?firmware hash} v
  [ "$(sha256sum $FW | cut -c1-16)" = "$WANT" ] || { echo "STOP: the firmware is not the pre-registered one"; return 1; }
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the stage changed since the pre-registration"; return 1; }
  [ "$(sha256sum $DRV_DIR/fpga_driver/run_baked_rungs_fpga.py | cut -c1-16)" = "$DRV_SHA" ] || { echo "STOP: the driver is not the pre-registered one"; return 1; }
  [ -e $D ] && { echo "STOP: $D exists; a run writes a fresh directory"; return 1; }
  mkdir -p $D; local L=$D/SEQ
  echo "TIER1_3R3_START $(date +%T) fw $WANT" >> $L
  cd $DRV_DIR || return 1
  FPGA_BITSTREAM=caplifive_vp2_bf3db3175.bit FPGA_URL="$(cat ${T1_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url})" FPGA_FW=$FW \
    BAKED_CTL=/test-domains/vmrun.sh BAKED_RUNGS="b0-stats b0-hello vm-contract" LADDER_FPGA_DIR=$T/oracles \
    BAKED_TIMEOUT=3600 BAKED_IDLE_S=200 PREFLIGHT=0 BAKED_WEDGE_APERTURES=1 BAKED_OUT=$D/board.txt \
    timeout 9000 ${T1_DRIVER:-python3 -m fpga_driver.run_baked_rungs_fpga} > $D/driver.log 2>&1 &
  local runner=$!
  ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420 bash board-watchdog.sh $D/driver.log 900 $runner > $D/watchdog.log 2>&1 &
  local wd=$!
  wait $runner; local rc=$?
  sleep 2; kill $wd 2>/dev/null; wait $wd 2>/dev/null
  echo "DRIVER rc=$rc $(date +%T); watchdog lines $(grep -c '^ALIVE\|^QUIET\|^STALE' $D/watchdog.log)" >> $L
  [ -s $D/board.txt ] || { echo "NO DATA: no board transcript" >> $L; return 2; }
  local r
  for r in b0-stats b0-hello vm-contract; do
    v=$(grep -a -o -E "^RESULT $r retval=-?[0-9]+" $D/board.txt | tail -1 | sed 's/.*retval=//')
    echo "RUNG $r retval=${v:-NONE} oracle=$(cat $T/oracles/$r.oracle) $([ "${v:-x}" = "$(cat $T/oracles/$r.oracle)" ] && echo AS-ORACLE || echo NOT-AS-ORACLE)" >> $L
  done
  python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/board-tier12/boot3/vm-contract-branch3r3.py $D/board.txt >> $L
  echo "TIER1_3R3_DONE $(date +%T)" >> $L
}
main "$@"; exit $?
