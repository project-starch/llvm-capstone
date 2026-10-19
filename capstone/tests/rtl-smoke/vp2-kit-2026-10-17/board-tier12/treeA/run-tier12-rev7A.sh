#!/bin/bash
# Tree A step 4 (2026-10-19): rev7's boot unchanged except the bitstream, caplifive_vp2c_8522d72fe. rev7 on bf3db3175 is
# its matched pair: the same stage, firmware and judge. Below, rev7's header.
# vp2 Tier 1 boot of module 39d8492d (the five-item revision), 2026-10-19: b0-stats, b0-hello, vm-contract (the module
# loaded with diag_emerg=1 node_flat_records=4096; the counters read before and after) and a vm-suite rung that runs
# sparse-churn LAST between two counter reads. Judged by rev7/rev7-judge.py after the rungs.
# Usage: run-tier12-rev7.sh <firmware sha256/16 as pre-registered>. T1_* variables are for the stubbed dry run only.
DRV_DIR=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke
DRV_SHA=fd38816a90c64c22
J=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/board-tier12
main() {
  set -u
  local T=/tmp/capstone/tier1 D=${T1_OUT:-/tmp/capstone/tier1/run3rev7A} FW=${T1_FW:-/tmp/capstone/tier1/fw3rev7/fw_payload.bin}
  local WANT=${1:?firmware hash} v r O=/tmp/capstone/tier1/oracles-rev7
  [ "$(sha256sum $FW | cut -c1-16)" = "$WANT" ] || { echo "STOP: the firmware is not the pre-registered one"; return 1; }
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the stage changed since the pre-registration"; return 1; }
  [ "$(sha256sum $T/stage/vm/capstone_vm.ko | cut -c1-16)" = 39d8492d5c18216b ] || { echo "STOP: the staged module is not 39d8492d"; return 1; }
  [ "$(sha256sum $DRV_DIR/fpga_driver/run_baked_rungs_fpga.py | cut -c1-16)" = "$DRV_SHA" ] || { echo "STOP: the driver is not the pre-registered one"; return 1; }
  [ -e $D ] && { echo "STOP: $D exists; a run writes a fresh directory"; return 1; }
  mkdir -p $D; local L=$D/SEQ
  echo "TIER12_REV7A_START $(date +%T) fw $WANT bitstream caplifive_vp2c_8522d72fe" >> $L
  cd $DRV_DIR || return 1
  FPGA_BITSTREAM=caplifive_vp2c_8522d72fe.bit FPGA_URL="$(cat ${T1_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url})" FPGA_FW=$FW \
    BAKED_CTL=/test-domains/vmrun.sh BAKED_RUNGS="b0-stats b0-hello vm-contract vm-suite" LADDER_FPGA_DIR=$O \
    BAKED_TIMEOUT=3600 BAKED_IDLE_S=200 PREFLIGHT=0 BAKED_WEDGE_APERTURES=1 BAKED_OUT=$D/board.txt \
    timeout 9000 ${T1_DRIVER:-python3 -m fpga_driver.run_baked_rungs_fpga} > $D/driver.log 2>&1 &
  local runner=$!
  ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420 bash board-watchdog.sh $D/driver.log 900 $runner > $D/watchdog.log 2>&1 &
  local wd=$!
  wait $runner; local rc=$?
  sleep 2; kill $wd 2>/dev/null; wait $wd 2>/dev/null
  echo "DRIVER rc=$rc $(date +%T); watchdog lines $(grep -c '^ALIVE\|^QUIET\|^STALE' $D/watchdog.log)" >> $L
  [ -s $D/board.txt ] || { echo "NO DATA: no board transcript" >> $L; return 2; }
  for r in b0-stats b0-hello vm-contract vm-suite; do
    v=$(grep -a -o -E "^RESULT $r retval=-?[0-9]+" $D/board.txt | tail -1 | sed 's/.*retval=//')
    echo "RUNG $r retval=${v:-NONE} oracle=$(cat $O/$r.oracle) $([ "${v:-x}" = "$(cat $O/$r.oracle)" ] && echo AS-ORACLE || echo NOT-AS-ORACLE)" >> $L
  done
  python3 $J/boot3/vm-contract-branch3r3.py $D/board.txt >> $L
  echo "FRTY retries $(grep -a -o 'FRTY:[0-9A-F]\{8\}' $D/board.txt | wc -l) at $(grep -a -o 'FRTY:[0-9A-F]\{8\}' $D/board.txt | sort | uniq -c | tr '\n' ' ')" >> $L
  echo "VM exits: $(grep -a -o '^VM_EXIT:[a-z_-]*:[0-9]*' $D/board.txt | tr '\n' ' ')" >> $L
  echo "VM_BEGIN without VM_EXIT: $(for b in $(grep -a -o '^VM_BEGIN:[a-z_-]*' $D/board.txt | cut -d: -f2); do grep -a -q "^VM_EXIT:$b:" $D/board.txt || echo $b; done | tr '\n' ' ')" >> $L
  python3 $J/rev7/rev7-judge.py $D/board.txt > $D/judge.txt 2>&1; echo "JUDGE rc=$? :: $(tail -1 $D/judge.txt)" >> $L
  sed 's/^/  /' $D/judge.txt >> $L
  python3 $J/boot3/cpmx-diag-read.py $D/board.txt > $D/cpmx-diag.txt 2>&1
  echo "CPMX-DIAG rc=$? :: $(tail -1 $D/cpmx-diag.txt)" >> $L
  echo "TIER12_REV7A_DONE $(date +%T)" >> $L
}
main "$@"; exit $?
