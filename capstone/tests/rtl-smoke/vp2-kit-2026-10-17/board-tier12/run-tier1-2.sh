#!/bin/bash
# vp2 Tier 1/2, boot 2: the v7 module and kit, the firmware on capstone-sbi vp2/tier1-monitor fba7812 (the S-trap
# injection + the v7 CSMINT routine gated to S). Rungs, judges and stage layout as boot 1 (run-tier1.sh); changed: the
# driver runs from the lane worktree (its classifier knows INJS/NODM/INJL), vm-contract's judge names the injected and
# monitor-stop branches, and the pins below are boot 2's.
# Usage: run-tier1-2.sh <firmware sha256/16 as pre-registered>. T1_* variables are for the stubbed dry run only.
# Filled in when the compiler lane's module arrives (STOP while any is still a placeholder):
RUNPY=/tmp/capstone/tier1/run-RUNPY_REV.py      # run.py at the module's revision
RUNPY_SHA=RUNPY_SHA16                           # its sha256/16
LLVM_BIN=LLVM_BIN_DIR                           # the llvm-nm run.py (in board-check.py) reads the .dom with
DRV_DIR=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke
DRV_SHA=4ae259caae50b35d                        # sha256/16 of the lane's run_baked_rungs_fpga.py (lane 0b8813639710)
main() {
  set -u
  local T=/tmp/capstone/tier1 D=${T1_OUT:-/tmp/capstone/tier1/run2} FW=${T1_FW:-/tmp/capstone/tier1/fw2/fw_payload.bin}
  local WANT=${1:?firmware hash} v
  for v in "$RUNPY" "$RUNPY_SHA" "$LLVM_BIN" "$DRV_SHA"; do
    case "$v" in *RUNPY_REV*|*RUNPY_SHA16*|*LLVM_BIN_DIR*|*DRV_SHA16*) echo "STOP: placeholder $v not filled"; return 1;; esac
  done
  [ "$(sha256sum $FW | cut -c1-16)" = "$WANT" ] || { echo "STOP: the firmware is not the pre-registered one"; return 1; }
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the stage changed since the pre-registration"; return 1; }
  [ "$(sha256sum $RUNPY | cut -c1-16)" = "$RUNPY_SHA" ] || { echo "STOP: run.py is not the module revision's"; return 1; }
  [ "$(sha256sum $DRV_DIR/fpga_driver/run_baked_rungs_fpga.py | cut -c1-16)" = "$DRV_SHA" ] || { echo "STOP: the driver is not the pre-registered one"; return 1; }
  [ -x "$LLVM_BIN/llvm-nm" ] || { echo "STOP: no llvm-nm in $LLVM_BIN (run.py reads the .dom with it)"; return 1; }
  [ -e $D ] && { echo "STOP: $D exists; a run writes a fresh directory"; return 1; }
  mkdir -p $D; local L=$D/SEQ
  echo "TIER1_2_START $(date +%T) fw $WANT" >> $L
  cd $DRV_DIR || return 1
  FPGA_BITSTREAM=caplifive_vp2_bf3db3175.bit FPGA_URL="$(cat ${T1_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url})" FPGA_FW=$FW \
    BAKED_CTL=/test-domains/vmrun.sh BAKED_RUNGS="b0-stats b0-hello vm-contract vm-suite" LADDER_FPGA_DIR=$T/oracles \
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
  for r in b0-stats b0-hello vm-contract vm-suite; do
    v=$(grep -a -o -E "^RESULT $r retval=-?[0-9]+" $D/board.txt | tail -1 | sed 's/.*retval=//')
    echo "RUNG $r retval=${v:-NONE} oracle=$(cat $T/oracles/$r.oracle) $([ "${v:-x}" = "$(cat $T/oracles/$r.oracle)" ] && echo AS-ORACLE || echo NOT-AS-ORACLE)" >> $L
  done
  python3 $T/vm-contract-branch.py $D/board.txt >> $L
  CAPSTONE_LLVM_BIN=$LLVM_BIN BOARD_EXPECT_FALSE=cleanup \
    python3 $T/board-check.py $D/board.txt $RUNPY $T/stage/vm/contract.dom --rung vm-suite > $D/vm-suite-check.txt 2>&1
  echo "VM-SUITE board-check rc=$? :: $(tail -1 $D/vm-suite-check.txt)" >> $L
  echo "TIER1_2_DONE $(date +%T)" >> $L
}
main "$@"; exit $?
