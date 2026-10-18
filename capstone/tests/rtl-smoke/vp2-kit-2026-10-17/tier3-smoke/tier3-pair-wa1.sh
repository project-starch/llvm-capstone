#!/bin/bash
# The S-27 paired boot (2026-10-18): the workaround module d71f207a3af5546b (resume CSRUNV with rd = x0, the kind read
# from frame +16) against 3005ed52e860c7af's records, in two boots that each differ from their old-module record in the
# module alone:
#   1. Tier 1 (b0-stats, b0-hello, vm-contract) through the Tier 1 recipe, firmware as pre-registered;
#      reference: ap1 (fw 83e5c5c6d6f89471, module 3005ed52).
#   2. pym0 through the batch session on t3-smoke/bkit7-fix6 (bkit7 with the module swapped), the same runner arguments
#      and images as pym0 on bkit7; reference: records/pym0.board.txt (0 of 20, cause 6 at 0x12).
# Boot 2 runs only if boot 1 reads as pre-registered: all three rungs as their oracles, vm-contract's identity branch,
# the banner once and 0 refusal lines. Otherwise it stops and says why.
# Usage: tier3-pair-wa1.sh <tier 1 firmware sha256/16> <out log>. Ends by itself.
# PAIR_T1 / PAIR_ARM / PAIR_T1_DIR / PAIR_ARM_DIR replace the two boots and their output dirs, and PAIR_TAG / PAIR_RESULTS
# the record tag and the results/ directory, for the stubbed dry run.
main() {
  set -u
  local FWH=${1:?tier 1 firmware hash} O=${2:?out log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone I=/mnt/hdd/$USER/capstone/t3-images-nb K=/mnt/hdd/$USER/capstone/t3-smoke/bkit7-fix6
  local LB=/tmp/capstone/vp2-prep/llvm-bin-v6 C=/tmp/capstone/wt-gate/capstone/bug-corpora
  local T1=${PAIR_T1:-$S/../board-tier12/boot3/run-tier12-wa1.sh} ARM=${PAIR_ARM:-$S/tier3-appvm-arm.sh}
  local D1=${PAIR_T1_DIR:-/tmp/capstone/tier1/run3wa1} R=${PAIR_ARM_DIR:-$H/t3-run/pym0-fix6} Q
  local TAG=${PAIR_TAG:-pym0-fix6} RD=${PAIR_RESULTS:-2026-10-18-vp2-rdx0}
  [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = d71f207a3af5546b ] || { echo "PAIR STOP: the kit's module is not d71f207a" >> "$O"; return 1; }
  pgrep -f "fpga_driver[.]run_baked" > /dev/null && { echo "PAIR STOP: a driver is running" >> "$O"; return 1; }
  echo "PAIR T1 BEGIN $(date +%T)" >> "$O"
  bash "$T1" "$FWH" >> "$O" 2>&1
  echo "PAIR T1 END rc=$? $(date +%T)" >> "$O"
  Q=$D1/SEQ
  [ -f "$Q" ] && grep -q "^TIER12_WA1_DONE" "$Q" || { echo "PAIR STOP: Tier 1 has no complete SEQ ($Q)" >> "$O"; return 1; }
  sed 's/^/  T1 /' "$Q" >> "$O"
  local bad=0 r
  for r in b0-stats b0-hello vm-contract; do grep -q "^RUNG $r retval=.* AS-ORACLE$" "$Q" || { echo "PAIR GATE: $r not as its oracle" >> "$O"; bad=1; }; done
  grep -q "^VM-CONTRACT BRANCH identity:" "$Q" || { echo "PAIR GATE: vm-contract is not the identity branch" >> "$O"; bad=1; }
  grep -q "^REFUSED-CSRUNV (must-fire) 0; banner 1$" "$Q" || { echo "PAIR GATE: refusal lines or banner not as expected: $(grep '^REFUSED-CSRUNV' "$Q")" >> "$O"; bad=1; }
  [ $bad = 0 ] || { echo "PAIR STOP: Tier 1 did not read as pre-registered; pym0 not booted $(date +%T)" >> "$O"; return 1; }
  pgrep -f "fpga_driver[.]run_baked" > /dev/null && { echo "PAIR STOP: a driver is running" >> "$O"; return 1; }
  echo "PAIR pym0-fix6 BEGIN $(date +%T)" >> "$O"
  EXPECT_IMAGES=$I/pym0.images.SHA256SUMS taskset -c 16-31 timeout 10800 bash "$ARM" --board-kit $K "$R" \
    $C/cpython/pymalloc-repros/runners/virtual/run-virtual.py --arm virtual-malloc --build $I/pym/cases-OFF \
    --controls-build $I/pym/controls-OFF --llvm-bin $LB >> "$O" 2>&1
  echo "PAIR pym0-fix6 END rc=$? $(date +%T)" >> "$O"
  KIT_DESC="kit bkit7-fix6 (module d71f207a3af5546b, the S-27 workaround: resume rd = x0), pin off" \
    RESULTS_DIR=$RD bash "$S/score-arm.sh" "$TAG" "$H/t3-pred-pym0/out" "$R" "$O" >> "$O" 2>&1
  echo "PAIR score rc=$?" >> "$O"
  # the compiler lane's twin on the new module's QEMU build (28882779c95b94ad), as a second reference
  [ -f "$H/t3-pym0-fix6/out/verdicts.jsonl" ] && python3 "$S/compare-pilot.py" "$H/t3-pym0-fix6/out" "$R/out" > "$R/compare-twin-fix6.txt" 2>&1
  echo "PAIR vs the fix6 twin rc=$? :: $(tail -1 "$R/compare-twin-fix6.txt" 2>/dev/null)" >> "$O"
  echo "PAIR DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
