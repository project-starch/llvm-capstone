#!/bin/bash
# Tree A, steps 4-6 of the plan (4ae54ec461da, amended f962be341d5d and 71f0ff3d2b38; the lead's own word 2026-10-19).
# They run on caplifive_vp2c_8522d72fe after steps 1-3 end with CHAIN_TREEA_STEPS123_DONE; any other end holds them.
#   4. Tier 1 on module 39d8492d: rev7's stage, firmware 7a53759a4773c81b and judge, with only the bitstream changed
#      (board-tier12/treeA/run-tier12-rev7A.sh). Its matched pair is rev7 on bf3db3175.
#   5. wm0, then wm1, on kit bkit7-rev2sA (bkit7-rev2s, module 39d8492d, budget x4, with the bitstream changed). These
#      are R-57's positive controls: each wedged on bf3db3175 on both modules.
#   6. pym0 on kit bkit7A (bkit7, module 3005ed52, with the bitstream changed), S-27's RTL fix two-sided: 0 of 20 on
#      bf3db3175 with the same images.
# Steps 5 and 6 run only if step 4's b0-stats, b0-hello and vm-contract read their oracles. Each arm uses its earlier
# runner arguments and images, is scored against the same prediction (t3-pred-<arm>/out), and is recorded as
# <arm>-treeA under results/2026-10-19-vp2c-treeA/.
# Usage: tier3-treeA-linux.sh <tree A CHAIN.log> <out> [arm...] (default wm0 wm1 pym0). Ends by itself. TAL_* replace
# the Tier 1 runner, the arm script, the scorer, the run root and the driver guard's pattern, for the stubbed dry run.
main() {
  set -u
  local W=${1:?tree A CHAIN.log} O=${2:?out}; shift 2; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone I=/mnt/hdd/$USER/capstone/t3-images-nb KS=/mnt/hdd/$USER/capstone/t3-smoke
  local LB=/tmp/capstone/vp2-prep/llvm-bin-v6 C=/tmp/capstone/wt-gate/capstone/bug-corpora G=$S/../board-tier12
  local T1=${TAL_T1:-$G/treeA/run-tier12-rev7A.sh} ARM=${TAL_ARM:-$S/tier3-appvm-arm.sh} SCORE=${TAL_SCORE:-$S/score-arm.sh}
  local R=${TAL_RUN:-$H/t3-run-treeA} RD=2026-10-19-vp2c-treeA D1=${TAL_T1_DIR:-/tmp/capstone/tier1/run3rev7A}
  local end=$(( $(date +%s) + 86400 )) t K P desc r v bad=0
  local -a arms args
  arms=("$@"); [ ${#arms[@]} -gt 0 ] || arms=(wm0 wm1 pym0)
  until grep -a -q "^CHAIN_TREEA_" "$W" 2>/dev/null || grep -a -q "^STOP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^CHAIN_TREEA_STEPS123_DONE" "$W" || { echo "TAL HOLD: steps 1-3 did not finish ($(tail -1 "$W"))" >> "$O"; return 1; }
  for K in $KS/bkit7A $KS/bkit7-rev2sA; do
    grep -qx "BOARD_BITSTREAM=caplifive_vp2c_8522d72fe" $K/board.env && ( cd $K && sha256sum -c --quiet SHA256SUMS ) \
      || { echo "TAL STOP: $K is not a tree A kit as recorded" >> "$O"; return 1; }
  done
  pgrep -f "${TAL_DRVPAT:-fpga_driver[.]run_baked}" > /dev/null && { echo "TAL STOP: a driver is running" >> "$O"; return 1; }
  echo "TAL STEP4 BEGIN $(date +%T)" >> "$O"
  bash "$T1" 7a53759a4773c81b >> "$O" 2>&1; echo "TAL STEP4 END rc=$? $(date +%T)" >> "$O"
  sed 's/^/  /' "$D1/SEQ" >> "$O" 2>/dev/null
  for r in b0-stats b0-hello vm-contract; do
    v=$(grep -a "^RUNG $r retval=" "$D1/SEQ" 2>/dev/null | tail -1)
    [[ "$v" == *" AS-ORACLE" ]] || bad=1
  done
  [ $bad = 0 ] || { echo "TAL STOP: step 4's Tier 1 rungs did not read their oracles; steps 5-6 do not run" >> "$O"; return 1; }
  for t in "${arms[@]}"; do
    case $t in
      wm0) K=$KS/bkit7-rev2sA desc="kit bkit7-rev2sA (module 39d8492d5c18216b; batch budget x4), tree A caplifive_vp2c_8522d72fe, pin off"
           args=($C/wireshark/wmem-repros/shared/run-defects.py --arm virtual-malloc --hosted-build $I/wmem/cases-OFF
                 --controls-hosted-build $I/wmem/controls-OFF --llvm-bin $LB); P=$H/t3-pred-wm0/out ;;
      wm1) K=$KS/bkit7-rev2sA desc="kit bkit7-rev2sA (module 39d8492d5c18216b; batch budget x4), tree A caplifive_vp2c_8522d72fe, pin off"
           args=($C/wireshark/wmem-repros/shared/run-defects.py --arm virtual-nested-pools --hosted-build $I/wmem/cases-ON
                 --controls-hosted-build $I/wmem/controls-ON --llvm-bin $LB); P=$H/t3-pred-wm1/out ;;
      pym0) K=$KS/bkit7A desc="kit bkit7A (module 3005ed52e860c7af, resume rd = x15), tree A caplifive_vp2c_8522d72fe, pin off"
            args=($C/cpython/pymalloc-repros/runners/virtual/run-virtual.py --arm virtual-malloc --build $I/pym/cases-OFF
                  --controls-build $I/pym/controls-OFF --llvm-bin $LB); P=$H/t3-pred-pym0/out ;;
      *) echo "TAL $t: no invocation for this arm" >> "$O"; continue ;;
    esac
    [ -f "$I/$t.images.SHA256SUMS" ] && [ -d "$P" ] || { echo "TAL $t: no image list or no prediction" >> "$O"; continue; }
    [ -e "$R/$t-treeA" ] && { echo "TAL $t: $R/$t-treeA exists" >> "$O"; continue; }
    pgrep -f "${TAL_DRVPAT:-fpga_driver[.]run_baked}" > /dev/null && { echo "TAL STOP: a driver is running" >> "$O"; return 1; }
    echo "T3APPVM $t-treeA BEGIN $(date +%T)" >> "$O"
    EXPECT_IMAGES=$I/$t.images.SHA256SUMS taskset -c 16-31 timeout 10800 bash "$ARM" --board-kit $K "$R/$t-treeA" "${args[@]}" >> "$O" 2>&1
    echo "T3APPVM $t-treeA END rc=$? $(date +%T)" >> "$O"
    KIT_DESC="$desc" RESULTS_DIR=$RD bash "$SCORE" "$t-treeA" "$P" "$R/$t-treeA" "$O" >> "$O" 2>&1
  done
  echo "TAL DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
