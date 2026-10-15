#!/bin/bash
# FP respin acceptance RE-RUN with the R-55 monitor fix (2026-10-15), on caplifive_fpb_afcd19bea.bit.
# The first run (/tmp/capstone/accept-fpb) stopped at R-43 a1: Linux hangs because the monitor's mtime capability
# reads msip (ISSUES R-55). Fix, both halves: caplifive-opensbi df0e9a8 / d7489e3 (cap_env_init's MMIO cursors) and
# capstone-sbi 6dad416 / fc5e3c6 (create_domain's globals copy, create_region, share_child_region); r55-fix/build-all-2.sh.
# Not re-run (bare, monitor-free, already recorded on this bitstream today): S-16 18/18, R-51, s10b, s1gap.
# Order: s06agg (the fixed firmware boots Linux: the cheap gate), R-43 a1..a10 (+ ladder, P1), C5u, boot C+F, boot W.
# Every step stops the chain on failure except the exploratory boot W. Body in main().
main() {
  set -u
  local D=/tmp/capstone/accept-fpb-r55 L=/tmp/capstone/accept-fpb-r55/SEQ F=/tmp/capstone/r55-fix/fw2
  local RS=$HOME/dev/llvm-capstone/capstone/tests/rtl-smoke
  local C=$HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot/components/opensbi
  export FPGA_BITSTREAM=${NEWBIT:?set NEWBIT to the console-reported name}
  echo "CHAIN_START $(date +%H:%M:%S) $FPGA_BITSTREAM" >> $L
  # Identities, before any boot.
  local want
  for want in "$F/fw-s06agg.bin b6cc1ea73168" "$F/c5u/fw_payload.bin a377c2415fd8" "$F/cf/fw_payload.bin d99e57bbee62" "$F/w/fw_payload.bin 76488c0374da"; do
    set -- $want
    [ "$(sha256sum $1 | cut -c1-12)" = "$2" ] || { echo "FW MISMATCH $1" >> $L; return 1; }
  done
  [ "$(git -C $C rev-parse --short=7 HEAD)" = df0e9a8 ] && grep -q "R-55" $C/lib/sbi/sbi_capstone_dom.c \
    || { echo "SHARED WRAPPER NOT AT df0e9a8" >> $L; return 1; }
  [ "$(git -C $C/lib/sbi/capstone-sbi rev-parse --short=7 HEAD)" = 6dad416 ] && grep -q "R-55" $C/lib/sbi/capstone-sbi/sbi_capstone.c \
    || { echo "SHARED MONITOR NOT AT 6dad416" >> $L; return 1; }
  echo "IDENTITIES OK $(date +%H:%M:%S)" >> $L

  # 1. s06agg + k800 on the fixed board-line firmware: Linux must boot (boot M's analogue with the production fix).
  mkdir -p $D/s06agg
  ( cd $RS
    export FPGA_URL="$(cat ~/.claude-kisp/secrets/fpga-console-url)" FPGA_FW=$F/fw-s06agg.bin PREFLIGHT=0
    export SQLITE_STAGE_OVERLAY=/tmp/capstone/s06agg-776/ref SQLITE_STAGE_TIMEOUT=300 SQLITE_IDLE_S=300 ENTRY_STALL_S=420 EARLY_HALT_CONTROL=0 WEDGE_TRACER=0 HALT_MUX_READS=0
    export K800_HASH=589ceee3853c6092 K800_ORACLES=$HOME/capstone-artifacts/k800-relinked-0x20000/orc PREFLIGHT_ORACLES=$HOME/capstone-artifacts/k800-relinked-0x20000/orc
    export SQLITE_STAGE_DOMS="/test-domains/lpc|k800:/test-domains/k800.dom,/test-domains/lpc|s06agg:/test-domains/s06agg.dom"
    export PROBE_SCOPED_OUT=$D/s06agg/boot.txt PROBE_RAW_OUT=$D/s06agg/boot-raw.txt
    timeout 1800 python3 -m fpga_driver.run_sqlite_stages_fpga > $D/s06agg/driver.log 2>&1 &
    local RUNNER=$!
    ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420 bash board-watchdog.sh "$D/s06agg/driver.log" 600 "$RUNNER" > $D/s06agg/watchdog.log 2>&1 &
    local WD=$!
    wait $RUNNER; local r=$?; kill $WD 2>/dev/null; wait $WD 2>/dev/null; exit $r )
  local rc=$?; echo "S06AGG rc=$rc $(date +%H:%M:%S)" >> $L
  [ $rc -eq 0 ] || { echo "STOPPED: the fixed firmware did not pass s06agg" >> $L; return 1; }
  sleep 8

  # 2. R-43 a1..a10 (+ ladder, P1), baked from the shared tree, whose wrapper is now df0e9a8.
  bash $D/r43/chain.sh; rc=$?
  echo "R43_CHAIN rc=$rc $(date +%H:%M:%S)" >> $L
  grep -q "R-55" $HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot/build-fpga/build/opensbi-custom/lib/sbi/sbi_capstone_dom.c \
    && grep -q "R-55" $HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot/build-fpga/build/opensbi-custom/lib/sbi/capstone-sbi/sbi_capstone.c \
    && echo "R43 BAKE CARRIED R-55" >> $L || { echo "R43 BAKE DID NOT CARRY R-55: its readings are on the unfixed monitor" >> $L; return 1; }
  [ $rc -eq 0 ] || { echo "STOPPED: the R-43 list failed" >> $L; return 1; }

  # 3. C5u: the supervised SQLite run on the CURRENT supcall monitor line with both R-55 halves and C5u's own payload
  #    (the 2026-10-03 monitor cannot be regenerated): a functional row; its overhead is reported, not compared.
  PREFLIGHT=0 SQLITE_STAGE_OVERLAY=/tmp/capstone/mon-c0/c5f-overlay SUPMON_BITSTREAM=$FPGA_BITSTREAM SUPMON_SPEEDTESTS=4 \
    bash /tmp/capstone/wt-b0/capstone/tests/rtl-smoke/supmon-2026-10-03/board-supmon.sh \
    supmon-c5u-r55 $F/c5u/fw_payload.bin > $D/c5u.log 2>&1
  echo "C5U rc=$? $(date +%H:%M:%S)" >> $L
  sleep 8

  # 4. Boot C+F on the fixed firmware (image 79f84cd81fef8dfa unchanged).
  ( cd /tmp/capstone/wt-b0/capstone/tests/rtl-smoke && FPGA_URL="$(cat ~/.claude-kisp/secrets/fpga-console-url)" \
    FPGA_FW=$F/cf/fw_payload.bin BAKED_CTL=/test-domains/b0run.sh \
    BAKED_RUNGS="b0-stats b0-hello b1-thread b2-memcached b3-setclock b3-oracle b3-fpstate b3-fpnow b3-fphold r54-bounds b3-clock b0-stats2" \
    LADDER_FPGA_DIR=$D/oracles BAKED_TIMEOUT=420 BAKED_IDLE_S=200 PREFLIGHT=0 BAKED_WEDGE_APERTURES=1 \
    BAKED_OUT=$D/bootCF.txt timeout 5400 python3 -m fpga_driver.run_baked_rungs_fpga > $D/bootCF.log 2>&1 )
  echo "BOOTCF rc=$? $(date +%H:%M:%S)" >> $L
  sleep 8

  # 5. EXPLORATORY: boot W, the globals-carve witness, on the fixed firmware. Predicted: cw-align 8, cw-pool 0.
  ( cd /tmp/capstone/wt-b0/capstone/tests/rtl-smoke && FPGA_URL="$(cat ~/.claude-kisp/secrets/fpga-console-url)" \
    FPGA_FW=$F/w/fw_payload.bin BAKED_CTL=/test-domains/b0run.sh BAKED_RUNGS="b0-stats b0-hello cw-align cw-pool b0-stats2" \
    LADDER_FPGA_DIR=$D/oracles BAKED_TIMEOUT=420 BAKED_IDLE_S=200 PREFLIGHT=0 BAKED_WEDGE_APERTURES=1 \
    BAKED_OUT=$D/bootW.txt timeout 2400 python3 -m fpga_driver.run_baked_rungs_fpga > $D/bootW.log 2>&1 )
  echo "BOOTW (exploratory) rc=$? $(date +%H:%M:%S)" >> $L
  echo "CHAIN_DONE $(date +%H:%M:%S)" >> $L
}
main "$@"; exit $?
