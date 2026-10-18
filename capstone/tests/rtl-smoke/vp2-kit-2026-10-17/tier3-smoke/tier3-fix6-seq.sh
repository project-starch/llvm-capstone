#!/bin/bash
# Tier 3 re-runs on the S-27 workaround kit bkit7-fix6 (module d71f207a3af5546b), after the paired boot: the
# run-virtual-cases arms that read S-27's fault on 3005ed52 (fso, pb12), then pb10 (S-23's data face, a kernel panic on
# 3005ed52; last, as the one expected not to return). Each is ARMS.spec's line under the tag <arm>-fix6, scored against
# the same prediction as its 3005ed52 run (<mnt-hdd>/capstone/t3-<arm>/bundle), its record naming the kit and its bundle
# filed under results/2026-10-18-vp2-rdx0/.
# It waits for the pair's end and runs only if the pair read pym0 at its verdict (20 of 20 cells the same); otherwise it
# HOLDS and says so. Usage: tier3-fix6-seq.sh <pair out> <out> [arm...] (default: fso pb12 pb10). Ends by itself.
# FIX6_SEQ / FIX6_SCORE / FIX6_RUN / FIX6_TAGSUFFIX / FIX6_RESULTS replace the sequence, the scorer, the run root, the
# tag suffix and the results directory for the stubbed dry run.
main() {
  set -u
  local P=${1:?pair out} O=${2:?out}; shift 2; local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone K=/mnt/hdd/$USER/capstone/t3-smoke/bkit7-fix6 SPEC=/mnt/hdd/$USER/capstone/t3-images-nb/ARMS.spec
  local SDK=/mnt/hdd/$USER/capstone/t3-freeze-nb/sdk LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  local SEQ=${FIX6_SEQ:-$S/tier3-seq.sh} SCORE=${FIX6_SCORE:-$S/score-arm.sh} R=${FIX6_RUN:-$H/t3-run-fix6}
  local SUF=${FIX6_TAGSUFFIX:--fix6} RD=${FIX6_RESULTS:-2026-10-18-vp2-rdx0} end=$(( $(date +%s) + 7200 )) a line
  local -a arms specs=()
  arms=("$@"); [ ${#arms[@]} -gt 0 ] || arms=(fso pb12 pb10)
  until grep -a -q "^PAIR DONE\|^PAIR STOP" "$P" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 20; done
  grep -a -q "^PAIR DONE" "$P" || { echo "FIX6SEQ HOLD: the pair did not finish ($(grep -a '^PAIR STOP' "$P" | tail -1))" >> "$O"; return 1; }
  grep -a -q "^pym0-fix6: 20 of 20 cells the same (rc 0)$" "$P" \
    || { echo "FIX6SEQ HOLD: the pair did not read pym0 at its verdict: $(grep -a '^pym0-fix6: ' "$P" | tail -1)" >> "$O"; return 1; }
  [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = d71f207a3af5546b ] || { echo "FIX6SEQ STOP: the kit's module is not d71f207a" >> "$O"; return 1; }
  for a in "${arms[@]}"; do
    line=$(grep "^$a|" "$SPEC") || { echo "FIX6SEQ STOP: $a is not in $SPEC" >> "$O"; return 1; }
    [ -d "$H/t3-$a/bundle" ] || { echo "FIX6SEQ STOP: no prediction bundle for $a" >> "$O"; return 1; }
    specs+=("$a$SUF|${line#*|}")
  done
  echo "FIX6SEQ BEGIN $(date +%T) arms ${arms[*]}" >> "$O"
  taskset -c 16-31 bash "$SEQ" "$K" "$R" "$SDK" "$LB" "${specs[@]}" >> "$O" 2>&1
  echo "FIX6SEQ SEQ END rc=$? $(date +%T)" >> "$O"
  for a in "${arms[@]}"; do
    grep -a -q "T3SEQ $a$SUF END" "$O" || { echo "FIX6SEQ $a$SUF not run" >> "$O"; continue; }
    KIT_DESC="kit bkit7-fix6 (module d71f207a3af5546b, the S-27 workaround: resume rd = x0), pin off" RESULTS_DIR=$RD \
      bash "$SCORE" "$a$SUF" "$H/t3-$a/bundle" "$R/$a$SUF" "$O" >> "$O" 2>&1
  done
  echo "FIX6SEQ DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
