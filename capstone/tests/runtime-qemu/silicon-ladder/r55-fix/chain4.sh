#!/bin/bash
# Attempt 4 of the FP respin re-run (R-55 fixed through split_out_cap's SHRINK, firmwares fw3). Attempt 3: s06agg and
# R-43 a1 PASSED (a1's readings identical to 0568f93a9's); a2 stopped at invocation 8 of 12 on a FULL region table
# (RGNF E00C, RGNN 0x60): under the full scheme split_out_cap's head/tail tests read DECODED bounds, so nearly every
# carve leaves an extra fragment (about 11 slots per invocation against 6 on 0568f93a9). That is understood and
# deterministic, so the remaining rows run, each its own cold boot, and do not stop on one another's failure.
# a2 re-runs after the region-table fix.
main() {
  set -u
  local D=/tmp/capstone/accept-fpb-r55 L=/tmp/capstone/accept-fpb-r55/SEQ F=/tmp/capstone/r55-fix/fw3 A=/tmp/capstone/accept-fpb-r55/r43
  local C=$HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot/components/opensbi
  export FPGA_BITSTREAM=${NEWBIT:?set NEWBIT}
  export PREFLIGHT_DTS_NODES=65536
  echo "CHAIN4_START $(date +%H:%M:%S) $FPGA_BITSTREAM" >> $L
  [ "$(git -C $C rev-parse --short=7 HEAD)" = df0e9a8 ] && [ "$(git -C $C/lib/sbi/capstone-sbi rev-parse --short=7 HEAD)" = 93440d5 ] \
    || { echo "SHARED TREE NOT AT df0e9a8 / 93440d5" >> $L; return 1; }
  local b
  for b in a3 a4 a5 a6; do bash $A/run-seq.sh $b; echo "R43 $b rc=$? $(date +%H:%M:%S)" >> $L; done
  bash $A/ladder-k0.sh; echo "R43 a7 ladder rc=$? $(date +%H:%M:%S)" >> $L
  bash $A/run-p1.sh c5 > $A/a8-p1c5-chain.log 2>&1; echo "R43 a8 rc=$? $(date +%H:%M:%S)" >> $L
  bash $A/run-p1.sh c6 > $A/a9-p1c6-chain.log 2>&1; echo "R43 a9 rc=$? $(date +%H:%M:%S)" >> $L
  bash $A/run-seq.sh a10; echo "R43 a10 rc=$? $(date +%H:%M:%S)" >> $L
  grep -q "R-55 (bounds)" $HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot/build-fpga/build/opensbi-custom/lib/sbi/capstone-sbi/sbi_capstone.c \
    && echo "R43 BAKE CARRIED R-55 (bounds)" >> $L || echo "R43 BAKE DID NOT CARRY R-55 (bounds): R-43 readings VOID" >> $L
  sleep 8
  PREFLIGHT=0 SQLITE_STAGE_OVERLAY=/tmp/capstone/mon-c0/c5f-overlay SUPMON_BITSTREAM=$FPGA_BITSTREAM SUPMON_SPEEDTESTS=4 \
    bash /tmp/capstone/wt-b0/capstone/tests/rtl-smoke/supmon-2026-10-03/board-supmon.sh supmon-c5u-r55 $F/c5u/fw_payload.bin > $D/c5u.log 2>&1
  echo "C5U rc=$? $(date +%H:%M:%S)" >> $L
  sleep 8
  ( cd /tmp/capstone/wt-b0/capstone/tests/rtl-smoke && FPGA_URL="$(cat ~/.claude-kisp/secrets/fpga-console-url)" \
    FPGA_FW=$F/cf/fw_payload.bin BAKED_CTL=/test-domains/b0run.sh \
    BAKED_RUNGS="b0-stats b0-hello b1-thread b2-memcached b3-setclock b3-oracle b3-fpstate b3-fpnow b3-fphold r54-bounds b3-clock b0-stats2" \
    LADDER_FPGA_DIR=$D/oracles BAKED_TIMEOUT=420 BAKED_IDLE_S=200 PREFLIGHT=0 BAKED_WEDGE_APERTURES=1 \
    BAKED_OUT=$D/bootCF.txt timeout 5400 python3 -m fpga_driver.run_baked_rungs_fpga > $D/bootCF.log 2>&1 )
  echo "BOOTCF rc=$? $(date +%H:%M:%S)" >> $L
  sleep 8
  ( cd /tmp/capstone/wt-b0/capstone/tests/rtl-smoke && FPGA_URL="$(cat ~/.claude-kisp/secrets/fpga-console-url)" \
    FPGA_FW=$F/w/fw_payload.bin BAKED_CTL=/test-domains/b0run.sh BAKED_RUNGS="b0-stats b0-hello cw-align cw-pool b0-stats2" \
    LADDER_FPGA_DIR=$D/oracles BAKED_TIMEOUT=420 BAKED_IDLE_S=200 PREFLIGHT=0 BAKED_WEDGE_APERTURES=1 \
    BAKED_OUT=$D/bootW.txt timeout 2400 python3 -m fpga_driver.run_baked_rungs_fpga > $D/bootW.log 2>&1 )
  echo "BOOTW (exploratory) rc=$? $(date +%H:%M:%S)" >> $L
  echo "CHAIN4_DONE $(date +%H:%M:%S)" >> $L
}
main "$@"; exit $?
