#!/bin/bash
# vp2 boot r9 (2026-10-18): the compiler lane's module ce3a542d00acce86 and run.py's suite with sparse-churn moved LAST (it
# wedged the core on r8); firmware on the retry monitor 007d81d. Rungs b0-stats, b0-hello, vm-contract, vm-suite last. board-check
# reads to the end of the transcript if vm-suite has no END marker (BOARD_TO_END=1), so a wedge in the last step still yields the rest.
#
#
# Usage: run-tier12-r9.sh <firmware sha256/16 as pre-registered>. T1_* variables are for the stubbed dry run only.
RUNPY=/tmp/capstone/tier1/run-457b78f27e10.py  # run.py at the module's revision (identical to the v9 kit's)
RUNPY_SHA=85461f0e25ab66bf
LLVM_BIN=/tmp/capstone/llvm-dev-current-20261017/bin
DRV_DIR=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke
DRV_SHA=4ae259caae50b35d
J=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/board-tier12/boot3
EXPECT_FALSE=threads,vm,heap-threads,threads_output,vm_contract,heap_threads,shared_heap_pages_released,cleanup
main() {
  set -u
  local T=/tmp/capstone/tier1 D=${T1_OUT:-/tmp/capstone/tier1/run3r9} FW=${T1_FW:-/tmp/capstone/tier1/fw3r9/fw_payload.bin}
  local WANT=${1:?firmware hash} v
  [ "$(sha256sum $FW | cut -c1-16)" = "$WANT" ] || { echo "STOP: the firmware is not the pre-registered one"; return 1; }
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the stage changed since the pre-registration"; return 1; }
  [ "$(sha256sum $RUNPY | cut -c1-16)" = "$RUNPY_SHA" ] || { echo "STOP: run.py is not the module revision's"; return 1; }
  [ "$(sha256sum $DRV_DIR/fpga_driver/run_baked_rungs_fpga.py | cut -c1-16)" = "$DRV_SHA" ] || { echo "STOP: the driver is not the pre-registered one"; return 1; }
  [ -x "$LLVM_BIN/llvm-nm" ] || { echo "STOP: no llvm-nm in $LLVM_BIN"; return 1; }
  [ -e $D ] && { echo "STOP: $D exists; a run writes a fresh directory"; return 1; }
  mkdir -p $D; local L=$D/SEQ
  echo "TIER12_R9_START $(date +%T) fw $WANT" >> $L
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
  python3 $J/vm-contract-branch3r3.py $D/board.txt >> $L
  CAPSTONE_LLVM_BIN=$LLVM_BIN BOARD_EXPECT_FALSE=$EXPECT_FALSE BOARD_TO_END=1 \
    python3 $J/board-check3.py $D/board.txt $RUNPY $T/stage/vm/contract.dom --rung vm-suite > $D/vm-suite-check.txt 2>&1
  echo "VM-SUITE board-check rc=$? :: $(tail -1 $D/vm-suite-check.txt)" >> $L
  echo "FRTY retries $(grep -a -o 'FRTY:[0-9A-F]\{8\}' $D/board.txt | wc -l) at $(grep -a -o 'FRTY:[0-9A-F]\{8\}' $D/board.txt | sort | uniq -c | tr '\n' ' ')" >> $L
  echo "DIAG fault repeats: $(grep -a -o 'DIAG fault repeat=[^ ]* cause=[0-9]* addr=[0-9a-f]* pc=[0-9a-f]* pte=[0-9a-f]*' $D/board.txt | sort | uniq -c | tr '\n' ';')" >> $L
  echo "ELOOP lines: $(grep -a -o 'capstone_vm: fault at 0x[0-9a-f]* cause [0-9]* resolved [0-9]* times' $D/board.txt | sort | uniq -c | tr '\n' ';')" >> $L
  python3 $J/cpmx-diag-read.py $D/board.txt > $D/cpmx-diag.txt 2>&1
  echo "CPMX-DIAG rc=$? :: $(tail -1 $D/cpmx-diag.txt)" >> $L
  echo "TIER12_R9_DONE $(date +%T)" >> $L
}
main "$@"; exit $?
