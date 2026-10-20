#!/bin/bash
# pb10's srz-nop pair on BOARD 2, tree B (2026-10-20; pre-registered in README.md before the boots): two cold boots on the
# pb10-srz kit family with ONLY capstone_vm.ko changed: pb10-srzc on the srz module 607164b9 (bkit7-pb10BZ, the control,
# rebuilt byte-identical by the compiler lane) and pb10-nop on 2a8f85f7 (bkit7-pb10BN: the three fused post-CSRUNV
# `csrw 0x5c1, x0` replaced by nop; compiler lane srz-nop-pair 5260ec650af4). Copied from tier3-pb10-srz-b2.sh.
# Usage: tier3-pb10-nop-b2.sh <chain out> <out>: starts once <chain out> carries CHAIN_PBNOP_GO. Ends by itself.
# PB_SEQ / PB_SCORE / PB_RUN / PB_BUSY stub the sequence, the scorer, the run root and the busy check for the dry run.
main() {
  set -u
  local W=${1:?chain out} O=${2:?out}; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone SPEC=/mnt/hdd/$USER/capstone/t3-images-nb/ARMS.spec
  local SDK=/mnt/hdd/$USER/capstone/t3-freeze-nb/sdk LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local SEQ=${PB_SEQ:-$S/tier3-seq.sh} SCORE=${PB_SCORE:-$S/score-arm.sh} R=${PB_RUN:-$H/t3-run-treeB} end=$(( $(date +%s) + 86400 ))
  local line pair kit tag mon corpus arm ad ban
  until grep -a -q "^CHAIN_PBNOP_GO\|^STOP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  grep -a -q "^CHAIN_PBNOP_GO" "$W" || { echo "PB10 HOLD: no CHAIN_PBNOP_GO ($(tail -1 "$W"))" >> "$O"; return 1; }
  line=$(grep "^pb10|" "$SPEC") || { echo "PB10 STOP: pb10 is not in $SPEC" >> "$O"; return 1; }
  [ -d "$H/t3-pb10/bundle" ] || { echo "PB10 STOP: no prediction bundle for pb10" >> "$O"; return 1; }
  local modw
  for pair in "bkit7-pb10BZ pb10-srzc f13dbc5 607164b9c1407c7c srevroot-zero-1" "bkit7-pb10BN pb10-nop f13dbc5 2a8f85f70d8efcdd srevroot-nopf-1"; do
    read -r kit tag mon modw ban <<<"$pair"; kit=$H/t3-smoke/$kit
    grep -qx "BOARD_BITSTREAM=caplifive_vp2c_8de3c48cf" $kit/board.env && grep -qx "FPGA_BOARD=2" $kit/board.env && grep -qx "BOARD_MONITOR_COMMIT=$mon" $kit/board.env \
      && grep -qx "BOARD_STEP_S=300" $kit/board.env && ( cd $kit && sha256sum -c --quiet SHA256SUMS ) \
      && [ "$(sha256sum $kit/adapter/module/capstone_vm.ko | cut -c1-16)" = "$modw" ] \
      || { echo "PB10 STOP: $kit is not the recorded tree B board-2 kit on monitor $mon" >> "$O"; return 1; }
    ${PB_BUSY:-python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2} > /dev/null || { echo "PB10 STOP: board 2 is busy" >> "$O"; return 1; }
    echo "PB10 $tag BEGIN $(date +%T) kit $(basename $kit) monitor $mon" >> "$O"
    taskset -c 16-31 bash "$SEQ" "$kit" "$R" "$SDK" "$LB" "$tag|${line#*|}" >> "$O" 2>&1
    echo "PB10 $tag SEQ END rc=$? $(date +%T)" >> "$O"
    local B=$R/$tag/raw/guest/board.txt
    echo "PB10 $tag S-23: staged module $(sha256sum $R/$tag/raw/stage/capstone_vm.ko 2>/dev/null | cut -c1-16) (want $modw); banner $ban x$(grep -a -c "$ban" $B 2>/dev/null); FRTY $(grep -a -c '^FRTY:' $B 2>/dev/null) DRT1 $(grep -a -c '^DRT1:' $B 2>/dev/null) DRTX $(grep -a -c '^DRTX:' $B 2>/dev/null) Oops $(grep -a -c Oops $B 2>/dev/null)" >> "$O"
    grep -a -q "T3SEQ $tag END" "$O" || { echo "PB10 $tag not run" >> "$O"; continue; }
    read -r corpus arm < <(python3 -c "import json,sys; j=json.load(open(sys.argv[1])); print(j['corpus'], j['arm'])" "$R/$tag/out/inputs.json" 2>/dev/null)
    ad=""; [ -n "${arm:-}" ] && ad=$arm-${tag#pb10-}   # filed as <arm>-fix and <arm>-treeA, never in place of each other
    KIT_DESC="kit $(basename $kit) (module $modw; batch budget 300 s per step; monitor capstone-sbi $mon), tree B caplifive_vp2c_8de3c48cf on board 2, pin off" \
      RESULTS_DIR=2026-10-20-vp2c-treeB ARM_DIR=$ad bash "$SCORE" "$tag" "$H/t3-pb10/bundle" "$R/$tag" "$O" >> "$O" 2>&1
  done
  echo "PB10 DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
