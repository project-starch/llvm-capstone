#!/bin/bash
# Boot BLb (pre-registered on lane vp2-tier12 before the boot): boot BL's image and rungs on the resident vp2, with the
# firmware of caplifive-opensbi capstone-bootstrap's capstone-sbi pointer bumped 234ee66 -> dafe4f2 (arm B), and, only
# if B differs from boot P, boot P's own firmware on vp2 (arm A), the arm that differs from B by the firmware alone.
# Usage: board-bootblb.sh B|A <firmware sha256/16 as pre-registered>. T_* variables are for the stubbed dry run only.
main() {
  set -u
  local ARM=${1:?B or A} WANT=${2:?firmware hash} FW
  case $ARM in
    B) FW=${T_FW:-/tmp/capstone/bootblb/fw/fw_payload.bin};;
    A) FW=${T_FW:-/tmp/capstone/bootp/fw/fw_payload.bin};;
    *) echo "STOP: arm is B or A"; return 1;;
  esac
  local D=${T_OUT:-/tmp/capstone/bootblb/run-$ARM} DRV=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke
  local P=/tmp/capstone/bootp/run/board-bootp.txt J=/tmp/capstone/bootblb/judge-blb.py
  [ "$(sha256sum $FW | cut -c1-16)" = "$WANT" ] || { echo "STOP: the firmware is not the pre-registered one"; return 1; }
  [ "$(sha256sum $DRV/fpga_driver/run_baked_rungs_fpga.py | cut -c1-16)" = 4ae259caae50b35d ] || { echo "STOP: the driver is not the pinned one"; return 1; }
  [ "$(sha256sum $P | cut -c1-16)" = "$(cat /tmp/capstone/bootblb/bootp-board.sha16)" ] || { echo "STOP: boot P's transcript changed"; return 1; }
  local C=$HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot/components/opensbi
  [ "$(git -C $C rev-parse --short=7 HEAD)" = df0e9a8 ] && [ "$(git -C $C/lib/sbi/capstone-sbi rev-parse --short=7 HEAD)" = 93440d5 ] \
    || { echo "STOP: the shared tree is not at df0e9a8 / 93440d5"; return 1; }
  [ -e $D ] && { echo "STOP: $D exists; a run writes a fresh directory"; return 1; }
  mkdir -p $D; local L=$D/SEQ
  echo "BOOTBLB_${ARM}_START $(date +%T) fw $WANT" >> $L
  cd $DRV || return 1
  FPGA_BITSTREAM=caplifive_vp2_bf3db3175.bit FPGA_URL="$(cat ${T_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url})" FPGA_FW=$FW \
    BAKED_CTL=/test-domains/b0run.sh BAKED_RUNGS="b0-stats b0-hello b5-hello b5-mruby-e-stats b5-mruby-stdin-stats b5-mruby-bench r54-bounds cw-pool b6-sublet b0-stats2" \
    LADDER_FPGA_DIR=/tmp/capstone/bootp/oracles BAKED_TIMEOUT=2700 BAKED_IDLE_S=200 PREFLIGHT=0 BAKED_WEDGE_APERTURES=1 \
    BAKED_OUT=$D/board-bootblb.txt timeout 9000 ${T_DRIVER:-python3 -m fpga_driver.run_baked_rungs_fpga} > $D/driver.log 2>&1 &
  local runner=$!
  ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420 bash board-watchdog.sh $D/driver.log 900 $runner > $D/watchdog.log 2>&1 &
  local wd=$!
  wait $runner; local rc=$?
  sleep 2; kill $wd 2>/dev/null; wait $wd 2>/dev/null
  echo "DRIVER rc=$rc $(date +%T); watchdog lines $(grep -c '^ALIVE\|^QUIET\|^STALE' $D/watchdog.log)" >> $L
  [ -s $D/board-bootblb.txt ] || { echo "NO DATA: no board transcript" >> $L; return 2; }
  python3 $J $P $D/board-bootblb.txt > $D/judge-vs-P.txt 2>&1
  echo "JUDGE $ARM vs P rc=$? :: $(tail -1 $D/judge-vs-P.txt)" >> $L
  local BREF=${T_BREF:-/tmp/capstone/bootblb/run-B/board-bootblb.txt}
  if [ $ARM = A ]; then
    [ -s $BREF ] || { echo "NO DATA: arm A needs arm B's transcript at $BREF" >> $L; return 2; }
    python3 $J $D/board-bootblb.txt $BREF > $D/judge-B-vs-A.txt 2>&1
    echo "JUDGE B vs A rc=$? :: $(tail -1 $D/judge-B-vs-A.txt)" >> $L
  fi
  echo "BOOTBLB_${ARM}_DONE $(date +%T)" >> $L
}
main "$@"; exit $?
