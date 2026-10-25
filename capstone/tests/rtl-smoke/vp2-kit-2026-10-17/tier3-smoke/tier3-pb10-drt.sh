#!/bin/bash
# pb10-drt on tree A (2026-10-19 evening; pre-registered in README.md before the boot): ffmpeg/pool, Sublet, whose case 01
# panicked on S-23's data face; one cold boot on kit bkit7-pb10D (capstone-sbi f13dbc5: the refusal read and S-23b step 1).
# Copied from tier3-pb10-rf.sh with the kit, tag and trigger changed.
# Usage: tier3-pb10-pair.sh <chain out> <out>: starts once <chain out> carries CHAIN_DRT_PB10_GO. Ends by itself.
# PB_SEQ / PB_SCORE / PB_RUN / PB_BUSY replace the sequence, the scorer, the run root and the board-1 busy check for the
# stubbed dry run.
main() {
  set -u
  local W=${1:?chain out} O=${2:?out}; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone SPEC=/mnt/hdd/$USER/capstone/t3-images-nb/ARMS.spec
  local SDK=/mnt/hdd/$USER/capstone/t3-freeze-nb/sdk LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local SEQ=${PB_SEQ:-$S/tier3-seq.sh} SCORE=${PB_SCORE:-$S/score-arm.sh} R=${PB_RUN:-$H/t3-run-treeA} end=$(( $(date +%s) + 86400 ))
  local line pair kit tag mon corpus arm ad
  until grep -a -q "^CHAIN_DRT_PB10_GO\|^STOP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  grep -a -q "^CHAIN_DRT_PB10_GO" "$W" || { echo "PB10 HOLD: no CHAIN_DRT_PB10_GO ($(tail -1 "$W"))" >> "$O"; return 1; }
  line=$(grep "^pb10|" "$SPEC") || { echo "PB10 STOP: pb10 is not in $SPEC" >> "$O"; return 1; }
  [ -d "$H/t3-pb10/bundle" ] || { echo "PB10 STOP: no prediction bundle for pb10" >> "$O"; return 1; }
  for pair in "bkit7-pb10D pb10-drt f13dbc5"; do
    read -r kit tag mon <<<"$pair"; kit=$H/t3-smoke/$kit
    grep -qx "BOARD_BITSTREAM=caplifive_vp2c_8522d72fe" $kit/board.env && grep -qx "BOARD_MONITOR_COMMIT=$mon" $kit/board.env \
      && grep -qx "BOARD_STEP_S=300" $kit/board.env && ( cd $kit && sha256sum -c --quiet SHA256SUMS ) \
      && [ "$(sha256sum $kit/adapter/module/capstone_vm.ko | cut -c1-16)" = d71f207a3af5546b ] \
      || { echo "PB10 STOP: $kit is not the recorded tree A kit on monitor $mon" >> "$O"; return 1; }
    ${PB_BUSY:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1} > /dev/null || { echo "PB10 STOP: board 1 is busy" >> "$O"; return 1; }
    echo "PB10 $tag BEGIN $(date +%T) kit $(basename $kit) monitor $mon" >> "$O"
    taskset -c 16-31 bash "$SEQ" "$kit" "$R" "$SDK" "$LB" "$tag|${line#*|}" >> "$O" 2>&1
    echo "PB10 $tag SEQ END rc=$? $(date +%T)" >> "$O"
    grep -a -q "T3SEQ $tag END" "$O" || { echo "PB10 $tag not run" >> "$O"; continue; }
    read -r corpus arm < <(python3 -c "import json,sys; j=json.load(open(sys.argv[1])); print(j['corpus'], j['arm'])" "$R/$tag/out/inputs.json" 2>/dev/null)
    ad=""; [ -n "${arm:-}" ] && ad=$arm-${tag#pb10-}   # filed as <arm>-fix and <arm>-treeA, never in place of each other
    KIT_DESC="kit $(basename $kit) (module d71f207a3af5546b, the S-27 workaround; batch budget 300 s per step; monitor capstone-sbi $mon), tree A caplifive_vp2c_8522d72fe, pin off" \
      RESULTS_DIR=2026-10-19-vp2c-treeA ARM_DIR=$ad bash "$SCORE" "$tag" "$H/t3-pb10/bundle" "$R/$tag" "$O" >> "$O" 2>&1
  done
  echo "PB10 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
