#!/bin/bash
# Re-run of fso, pb12 and pb10 on the S-27 workaround module with a batch budget sized for programs that complete
# (2026-10-18). fso-fix6 on bkit7-fix6 ran out of its batch rung's wait (BAKED_TIMEOUT = the runner's 300 + 60 s per
# step = 1500 s for 20 steps; on d71f207a each step runs ~2 min instead of faulting in seconds), so the sequence stopped
# before pb12 and pb10. Kit bkit7-fix6t is bkit7-fix6 with BOARD_STEP_S=300 (board-run.sh scales the per-step part:
# 300 + 300 s per step); the module, launcher, recipe otherwise, runner and images are unchanged. Tags <arm>-fix6b,
# scored against the same predictions, bundles under results/2026-10-18-vp2-rdx0/ (a re-run whose arm already has a
# bundle there is filed as <arm>-b by ARM_DIR).
# Runs after the fence discriminator (FENCE DONE or STOP). An arm whose -fix6b run already ended completed is skipped.
# Usage: tier3-fix6b-seq.sh <fence out> <out> [arm...] (default: fso pb12 pb10). Ends by itself.
# FIX6B_SEQ / FIX6B_SCORE / FIX6B_RUN / FIX6B_TAGSUFFIX / FIX6B_RESULTS replace the sequence, the scorer, the run root,
# the tag suffix and the results directory for the stubbed dry run.
main() {
  set -u
  local P=${1:?fence out} O=${2:?out}; shift 2; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone K=/mnt/hdd/$USER/capstone/t3-smoke/bkit7-fix6t SPEC=/mnt/hdd/$USER/capstone/t3-images-nb/ARMS.spec
  local SDK=/mnt/hdd/$USER/capstone/t3-freeze-nb/sdk LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local SEQ=${FIX6B_SEQ:-$S/tier3-seq.sh} SCORE=${FIX6B_SCORE:-$S/score-arm.sh} R=${FIX6B_RUN:-$H/t3-run-fix6}
  local SUF=${FIX6B_TAGSUFFIX:--fix6b} RD=${FIX6B_RESULTS:-2026-10-18-vp2-rdx0} end=$(( $(date +%s) + 43200 )) a line
  local -a arms specs=()
  arms=("$@"); [ ${#arms[@]} -gt 0 ] || arms=(fso pb12 pb10)
  until grep -a -q "^FENCE DONE\|^FENCE STOP" "$P" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 30; done
  grep -a -q "^FENCE DONE\|^FENCE STOP" "$P" || { echo "FIX6BSEQ HOLD: the fence arm did not end" >> "$O"; return 1; }
  [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = d71f207a3af5546b ] || { echo "FIX6BSEQ STOP: the kit's module is not d71f207a" >> "$O"; return 1; }
  grep -qx "BOARD_STEP_S=300" $K/board.env || { echo "FIX6BSEQ STOP: the kit does not set BOARD_STEP_S=300" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) || { echo "FIX6BSEQ STOP: the kit does not match its SHA256SUMS" >> "$O"; return 1; }
  for a in "${arms[@]}"; do
    line=$(grep "^$a|" "$SPEC") || { echo "FIX6BSEQ STOP: $a is not in $SPEC" >> "$O"; return 1; }
    [ -d "$H/t3-$a/bundle" ] || { echo "FIX6BSEQ STOP: no prediction bundle for $a" >> "$O"; return 1; }
    grep -a -q "virtual boot: completed" "$R/$a$SUF/run.log" 2>/dev/null && { echo "FIX6BSEQ $a$SUF already completed; skipped" >> "$O"; continue; }
    specs+=("$a$SUF|${line#*|}")
  done
  [ ${#specs[@]} -gt 0 ] || { echo "FIX6BSEQ DONE $(date +%T) (nothing to run)" >> "$O"; return 0; }
  echo "FIX6BSEQ BEGIN $(date +%T) specs ${#specs[@]}" >> "$O"
  taskset -c 16-31 bash "$SEQ" "$K" "$R" "$SDK" "$LB" "${specs[@]}" >> "$O" 2>&1
  echo "FIX6BSEQ SEQ END rc=$? $(date +%T)" >> "$O"
  local ad corpus arm
  for a in "${arms[@]}"; do
    grep -a -q "T3SEQ $a$SUF END" "$O" || { echo "FIX6BSEQ $a$SUF not run" >> "$O"; continue; }
    read -r corpus arm < <(python3 -c "import json,sys; j=json.load(open(sys.argv[1])); print(j['corpus'], j['arm'])" "$R/$a$SUF/out/inputs.json" 2>/dev/null)
    ad=""; [ -n "${arm:-}" ] && [ -e "$S/../../../../bug-corpora/$corpus/results/$RD/$arm" ] && ad=$arm-b
    KIT_DESC="kit bkit7-fix6t (module d71f207a3af5546b, the S-27 workaround; batch budget 300 s per step), pin off" \
      RESULTS_DIR=$RD ARM_DIR=$ad bash "$SCORE" "$a$SUF" "$H/t3-$a/bundle" "$R/$a$SUF" "$O" >> "$O" 2>&1
  done
  echo "FIX6BSEQ DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
