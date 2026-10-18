#!/bin/bash
# vp2 Tier 1/2 on the board: one boot of the private image (BL's image + /test-domains/vm, flat module 0d62a1bcdeecc13f,
# kit c14933904de4), rungs b0-stats, b0-hello (controls), vm-contract (Tier 1, the virtual identity reading), vm-suite
# (Tier 2, run.py's steps). Then the judges: each rung's RESULT against its oracle; vm-contract's branch; vm-suite by
# board-check.py (run.py's own analysis block, c14933904de4) with BOARD_EXPECT_FALSE=cleanup.
# Usage: run-tier1.sh <firmware sha256/16 as pre-registered>. T1_* variables are for the stubbed dry run only.
main() {
  set -u
  local T=/tmp/capstone/tier1 D=${T1_OUT:-/tmp/capstone/tier1/run} FW=/tmp/capstone/tier1/fw/fw_payload.bin
  local WANT=${1:?firmware hash} RUNPY=$T/run-c14933904de4.py
  [ "$(sha256sum $FW | cut -c1-16)" = "$WANT" ] || { echo "STOP: the firmware is not the pre-registered one"; return 1; }
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "STOP: the stage changed since the pre-registration"; return 1; }
  [ "$(sha256sum $RUNPY | cut -c1-16)" = d03ea74cc714f17a ] || { echo "STOP: run.py is not c14933904de4's"; return 1; }
  [ -e $D ] && { echo "STOP: $D exists; a run writes a fresh directory"; return 1; }
  mkdir -p $D; local L=$D/SEQ
  echo "TIER1_START $(date +%T) fw $WANT" >> $L
  cd /tmp/capstone/wt-b0/capstone/tests/rtl-smoke || return 1
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
  local r v
  for r in b0-stats b0-hello vm-contract vm-suite; do
    v=$(grep -a -o -E "^RESULT $r retval=-?[0-9]+" $D/board.txt | tail -1 | sed 's/.*retval=//')
    echo "RUNG $r retval=${v:-NONE} oracle=$(cat $T/oracles/$r.oracle) $([ "${v:-x}" = "$(cat $T/oracles/$r.oracle)" ] && echo AS-ORACLE || echo NOT-AS-ORACLE)" >> $L
  done
  # vm-contract's branches (pre-registered): 29 at contract.dom's FIRST allocation, before VIRTUAL_APPLICATION_OK, is
  # "header or layout, kit side"; a 25 on the stale access after VIRTUAL_APPLICATION_OK is the identity reading.
  python3 - $D/board.txt >> $L <<'EOF'
import re, sys
t = open(sys.argv[1], errors="replace").read().replace("\r", "")
m = list(re.finditer(r"^### RUNG \d+/\d+ vm-contract ###$(.*?)^### RUNG \d+ vm-contract END rc=-?\d+ ###$", t, re.S | re.M))
if len(m) != 1: print(f"VM-CONTRACT NO DATA: {len(m)} segments"); sys.exit()
s = m[0].group(1); ok = "VIRTUAL_APPLICATION_OK" in s
pre = s.split("VIRTUAL_APPLICATION_OK")[0] if ok else s
if not ok and "domain fault cause=29" in pre: print("VM-CONTRACT BRANCH layout: 29 before VIRTUAL_APPLICATION_OK (kit side; to the RTL lane with the table dump)")
elif ok and re.search(r"VM_EXIT:stale:139", s) and "domain fault cause=25" in s: print("VM-CONTRACT BRANCH identity: VIRTUAL_APPLICATION_OK, then the stale access denied with 25 (exit 139)")
else: print(f"VM-CONTRACT BRANCH other: ok-line={ok} exits={re.findall(r'VM_EXIT:\S+', s)} causes={re.findall(r'cause=\d+', s)}")
EOF
  CAPSTONE_LLVM_BIN=/tmp/capstone/vp2-prep/llvm-bin-v6 BOARD_EXPECT_FALSE=cleanup \
    python3 $T/board-check.py $D/board.txt $RUNPY $T/stage/vm/contract.dom --rung vm-suite > $D/vm-suite-check.txt 2>&1
  echo "VM-SUITE board-check rc=$? :: $(tail -1 $D/vm-suite-check.txt)" >> $L
  echo "TIER1_DONE $(date +%T)" >> $L
}
main "$@"; exit $?
