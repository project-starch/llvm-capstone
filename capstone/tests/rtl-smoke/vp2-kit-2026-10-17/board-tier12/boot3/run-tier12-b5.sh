#!/bin/bash
# vp2 Tier 2 boot 5 (2026-10-18), the arena pin: module 3005ed52e860c7af (arena-pin-1 on a2a0b2f4) with vm-suite-inner-b5.sh: arena_pin=1 for
# the whole suite (read back), resolve_dirty on for vm only (read back), the kit applications after every returning step (VM_BEGIN
# markers name the step a wedge silences), the sparse pair NOT RUN (S-25). The module's lines come from the KMSG- dmesg reads.
# reads to the end of the transcript if vm-suite has no END marker (BOARD_TO_END=1), so a wedge in the last step still yields the rest.
#
#
# Usage: run-tier12-r11.sh <firmware sha256/16 as pre-registered>. T1_* variables are for the stubbed dry run only.
RUNPY=/tmp/capstone/tier1/run-457b78f27e10.py  # run.py at the module's revision (identical to the v9 kit's)
RUNPY_SHA=85461f0e25ab66bf
LLVM_BIN=/tmp/capstone/llvm-dev-current-20261017/bin
DRV_DIR=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke
DRV_SHA=fd38816a90c64c22
J=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/board-tier12/boot3
# The false set: the compiler lane's QEMU record of run.py under arena_pin=1 (b290f565, svadu=false, module 5bfc38f8 =
# 3005ed52's source), with the sparse pair's lines cut from its serial (b5-prediction.txt) -- 15 -- plus cleanup (the board
# kernel cannot unload the module).
EXPECT_FALSE=threads,vm,heap-threads,sparse-churn,sparse-stale,threads_output,fault_causes,vm_denial_sites,vm_contract,heap_threads,vm_denial_operations,linux_pages_released,linux_demand_faults,shared_heap_pages_released,sparse_recycling,cleanup
main() {
  set -u
  local T=/tmp/capstone/tier1 D=${T1_OUT:-/tmp/capstone/tier1/run3b5} FW=${T1_FW:-/tmp/capstone/tier1/fw3b5/fw_payload.bin}
  local WANT=${1:?firmware hash} v
  [ "$(sha256sum $FW | cut -c1-16)" = "$WANT" ] || { echo "STOP: the firmware is not the pre-registered one"; return 1; }
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the stage changed since the pre-registration"; return 1; }
  [ "$(sha256sum $RUNPY | cut -c1-16)" = "$RUNPY_SHA" ] || { echo "STOP: run.py is not the module revision's"; return 1; }
  [ "$(sha256sum $DRV_DIR/fpga_driver/run_baked_rungs_fpga.py | cut -c1-16)" = "$DRV_SHA" ] || { echo "STOP: the driver is not the pre-registered one"; return 1; }
  [ -x "$LLVM_BIN/llvm-nm" ] || { echo "STOP: no llvm-nm in $LLVM_BIN"; return 1; }
  [ -e $D ] && { echo "STOP: $D exists; a run writes a fresh directory"; return 1; }
  mkdir -p $D; local L=$D/SEQ
  echo "TIER12_B5_START $(date +%T) fw $WANT" >> $L
  cd $DRV_DIR || return 1
  FPGA_BITSTREAM=caplifive_vp2_bf3db3175.bit FPGA_URL="$(cat ${T1_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url})" FPGA_FW=$FW \
    BAKED_CTL=/test-domains/vmrun.sh BAKED_RUNGS="b0-stats b0-hello vm-contract vm-suite" LADDER_FPGA_DIR=$T/oracles-b5 \
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
    echo "RUNG $r retval=${v:-NONE} oracle=$(cat $T/oracles-b5/$r.oracle) $([ "${v:-x}" = "$(cat $T/oracles-b5/$r.oracle)" ] && echo AS-ORACLE || echo NOT-AS-ORACLE)" >> $L
  done
  python3 $J/vm-contract-branch3r3.py $D/board.txt >> $L
  CAPSTONE_LLVM_BIN=$LLVM_BIN BOARD_EXPECT_FALSE=$EXPECT_FALSE BOARD_TO_END=1 \
    python3 $J/board-check3.py $D/board.txt $RUNPY $T/stage/vm/contract.dom --rung vm-suite > $D/vm-suite-check.txt 2>&1
  echo "VM-SUITE board-check rc=$? :: $(tail -1 $D/vm-suite-check.txt)" >> $L
  echo "FRTY retries $(grep -a -o 'FRTY:[0-9A-F]\{8\}' $D/board.txt | wc -l) at $(grep -a -o 'FRTY:[0-9A-F]\{8\}' $D/board.txt | sort | uniq -c | tr '\n' ' ')" >> $L
  echo "DIAG fault repeats: $(grep -a -o 'DIAG fault repeat=[^ ]* cause=[0-9]* addr=[0-9a-f]* pc=[0-9a-f]* pte=[0-9a-f]*' $D/board.txt | sort | uniq -c | tr '\n' ';')" >> $L
  echo "ELOOP lines: $(grep -a -o 'capstone_vm: fault at 0x[0-9a-f]* cause [0-9]* resolved [0-9]* times' $D/board.txt | sort | uniq -c | tr '\n' ';')" >> $L
  echo "BANNER (console, pr_emerg; KMSG copies excluded): $(grep -a -v '^KMSG-' $D/board.txt | grep -a -o 'capstone_vm: loaded, diag [a-z0-9-]*, supervisor_slots=[0-9]*' | sort | uniq -c | tr '\n' ';')" >> $L
  echo "RESOLVE_DIRTY read-back: $(grep -a -o 'RESOLVE_DIRTY:[0-9YN]*:[a-z0-9-]*' $D/board.txt | tr '\n' ' ')" >> $L
  echo "ARENA_PIN read-back: $(grep -a -o 'ARENA_PIN:[0-9YN]*:[a-z0-9-]*' $D/board.txt | tr '\n' ' ')" >> $L
  echo "NOT RUN (S-25): $(grep -a -o '^VM_NOT_RUN:[a-z_-]*:S-25' $D/board.txt | tr '\n' ' ')" >> $L
  echo "BEGUN (kit applications): $(grep -a -o '^VM_BEGIN:[a-z_]*' $D/board.txt | tr '\n' ' ')" >> $L
  echo "KIT results: $(grep -a -o '^SQLITE_SUM=[0-9]*\|^SQLITE_PERSIST=[0-9]*\|^MRUBY_SUM=[0-9]*\|^MRUBY_FILE=[a-z]*' $D/board.txt | tr '\n' ' ')" >> $L
  local w
  for w in VM RETURNING APPS; do
    echo "KMSG window $w: $(grep -a -c "^KMSG-AFTER-$w: " $D/board.txt) lines, END $(grep -a -c "^KMSG-AFTER-$w-END" $D/board.txt); thread-refused (must-fire) $(grep -a -c "^KMSG-AFTER-$w: .*capstone_vm: thread create refused" $D/board.txt); bound lines $(grep -a -c "^KMSG-AFTER-$w: .*capstone_vm: fault at .* resolved" $D/board.txt); refusal sites $(tr -d '\r' < $D/board.txt | grep -a "^KMSG-AFTER-$w: " | grep -a -o 'resolve refused at [a-z]*' | sort | uniq -c | tr '\n' ' ')" >> $L
  done
  echo "VM exits: $(grep -a -o '^VM_EXIT:[a-z_-]*:[0-9]*' $D/board.txt | tr '\n' ' ')" >> $L
  python3 $J/cpmx-diag-read.py $D/board.txt > $D/cpmx-diag.txt 2>&1
  echo "CPMX-DIAG rc=$? :: $(tail -1 $D/cpmx-diag.txt)" >> $L
  echo "TIER12_B5_DONE $(date +%T)" >> $L
}
main "$@"; exit $?
