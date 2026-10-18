#!/bin/bash
# Tier 3 appvm arms on the S-27 workaround kit bkit7-fix6 (module d71f207a3af5546b), after tier3-fix6-seq.sh: pym1 and
# the sq0 subsets S-27 hit on 3005ed52 (b, c, d, e), then wm0 and wm1 last (each wedged on 3005ed52). Each arm is one
# boot through tier3-appvm-arm.sh with its 3005ed52 run's runner arguments, its collected images gated against its
# t3-images-nb list, and is scored against the same prediction as its 3005ed52 run (t3-pred-<arm>/out, sq0's cut to the
# subset by subset-bundle.py), the record naming the kit, the bundle under results/2026-10-18-vp2-rdx0/.
# It waits for the fix6 sequence's end and runs only after FIX6SEQ DONE (a HOLD or STOP there holds this too).
# Usage: tier3-fix6-appvm.sh <fix6 seq out> <out> [arm...] (default: pym1 sq0b sq0c sq0d sq0e wm0 wm1). Ends by itself.
# FIX6A_ARM / FIX6A_SCORE / FIX6A_RUN / FIX6A_TAGSUFFIX / FIX6A_RESULTS / FIX6A_SUBSETS replace the arm script, the
# scorer, the run root, the tag suffix, the results directory and the subset-prediction root for the stubbed dry run;
# FIX6A_DRVPAT the running-driver guard's pattern (a dry run beside a live boot would otherwise stop on it, correctly).
main() {
  set -u
  local W=${1:?fix6 seq out} O=${2:?out}; shift 2; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone I=/mnt/hdd/$USER/capstone/t3-images-nb K=/mnt/hdd/$USER/capstone/t3-smoke/bkit7-fix6
  local LB=/tmp/capstone/vp2-prep/llvm-bin-v6 C=/tmp/capstone/wt-gate/capstone/bug-corpora
  local ARM=${FIX6A_ARM:-$S/tier3-appvm-arm.sh} SCORE=${FIX6A_SCORE:-$S/score-arm.sh} R=${FIX6A_RUN:-$H/t3-run-fix6}
  local SUF=${FIX6A_TAGSUFFIX:--fix6} RD=${FIX6A_RESULTS:-2026-10-18-vp2-rdx0} SUB=${FIX6A_SUBSETS:-$H/t3-subsets-fix6}
  local end=$(( $(date +%s) + 21600 )) t tag P only rc
  local -a arms args
  arms=("$@"); [ ${#arms[@]} -gt 0 ] || arms=(pym1 sq0b sq0c sq0d sq0e wm0 wm1)
  until grep -a -q "^FIX6SEQ DONE\|^FIX6SEQ HOLD\|^FIX6SEQ STOP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  grep -a -q "^FIX6SEQ DONE" "$W" || { echo "FIX6APPVM HOLD: the fix6 sequence did not finish ($(grep -a '^FIX6SEQ' "$W" | tail -1))" >> "$O"; return 1; }
  [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = d71f207a3af5546b ] || { echo "FIX6APPVM STOP: the kit's module is not d71f207a" >> "$O"; return 1; }
  echo "FIX6APPVM BEGIN $(date +%T) arms ${arms[*]}" >> "$O"
  for t in "${arms[@]}"; do
    tag=$t$SUF
    case $t in
      pym1) args=($C/cpython/pymalloc-repros/runners/virtual/run-virtual.py --arm virtual-nested-pools --build $I/pym/cases-ON
                  --controls-build $I/pym/controls-ON --llvm-bin $LB); P=$H/t3-pred-pym1/out ;;
      sq0?) only=$(cut -c67-68 "$I/$t.images.SHA256SUMS" | grep -v '^co' | sort -u | paste -sd,)
            args=($C/sqlite/engine-repros/shared/run-virtual.py --arm virtual-malloc --build $I/sqlite/build-virtual-malloc
                  --only "$only" --llvm-bin $LB); P=$SUB/$t ;;
      wm0) args=($C/wireshark/wmem-repros/shared/run-defects.py --arm virtual-malloc --hosted-build $I/wmem/cases-OFF
                 --controls-hosted-build $I/wmem/controls-OFF --llvm-bin $LB); P=$H/t3-pred-wm0/out ;;
      wm1) args=($C/wireshark/wmem-repros/shared/run-defects.py --arm virtual-nested-pools --hosted-build $I/wmem/cases-ON
                 --controls-hosted-build $I/wmem/controls-ON --llvm-bin $LB); P=$H/t3-pred-wm1/out ;;
      *) echo "FIX6APPVM $t: no invocation for this arm" >> "$O"; continue ;;
    esac
    [ -f "$I/$t.images.SHA256SUMS" ] || { echo "FIX6APPVM $t: no image list" >> "$O"; continue; }
    pgrep -f "${FIX6A_DRVPAT:-fpga_driver[.]run_baked}" > /dev/null && { echo "FIX6APPVM STOP: a driver is running" >> "$O"; return 1; }
    echo "T3APPVM $tag BEGIN $(date +%T)" >> "$O"
    EXPECT_IMAGES=$I/$t.images.SHA256SUMS taskset -c 16-31 timeout 10800 bash "$ARM" --board-kit $K "$R/$tag" "${args[@]}" >> "$O" 2>&1
    echo "T3APPVM $tag END rc=$? $(date +%T)" >> "$O"
    if [ "${t#sq0}" != "$t" ]; then
      python3 "$S/subset-bundle.py" "$H/t3-pred-sq0/out" "$R/$tag/out" "$SUB/$t" >> "$O" 2>&1 || { echo "FIX6APPVM $tag: subset cut failed" >> "$O"; continue; }
    fi
    local ad=""; [ "${t#sq0}" != "$t" ] && ad=vp2-malloc-$t
    KIT_DESC="kit bkit7-fix6 (module d71f207a3af5546b, the S-27 workaround: resume rd = x0), pin off" RESULTS_DIR=$RD ARM_DIR=$ad \
      bash "$SCORE" "$tag" "$P" "$R/$tag" "$O" >> "$O" 2>&1
  done
  echo "FIX6APPVM DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
