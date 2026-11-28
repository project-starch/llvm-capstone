#!/bin/bash
# chain-dc.sh (nested lane, 2026-11-26): the v3 traces. gcc first (B5TL1 502.gcc_r test, small): graded, then read by
# rcm-lb-dc at 64x4 and at 1024x8; it must show v3 mismatch_vs_LB0 0, bit44 > 0 and a PASS stage, or mcf does not run.
# (2026-11-27: CHAIN_BENCH=mcf:505.mcf_r runs mcf alone, after a gcc run whose gate failed only on the window) and dc-window.py must read WINDOW-OK on its console. Then mcf (505.mcf_r test, the stage of v2's spec-mcf.2201746), read the same way. Writes chain-dc.log, CHAIN-DC-DONE.
main() {
  local R=<hdd>/capstone/nested/rectrace-dc X=<hdd>/capstone/b5b2linux BS=<hdd>/capstone/c4/b5b2spec/board-stages
  cd $R || return 2; local L=$R/chain-dc.log
  for e in ${CHAIN_BENCH:-gcc:502.gcc_r mcf:505.mcf_r}; do
    local n=${e%%:*} b=${e#*:} D=$R/runs/spec-${e%%:*}
    echo "$(date '+%F %T') begin $n" >> $L
    bash $R/trace-spec-dc.sh $n $BS/B5TL1/$b >> $L 2>&1; echo "$(date '+%F %T') $n rc=$?" >> $L
    python3 $R/dc-window.py $D/boot.log $D/snap.txt $D/dcplugin.txt > $D/window.out 2>&1; echo "$n window rc=$? $(tail -1 $D/window.out)" >> $L
    SPEC_STAGE_ROOT=$BS python3 $X/grade-spec-chain-v2.py $D/boot.log $D/grade B5TL1-$b > $D/grade.out 2>&1; echo "$n grade rc=$? $(tail -1 $D/grade.out)" >> $L
    local T; T=$(ls $R/traces/spec-$n.* 2>/dev/null | command grep -av stats)
    [ "$(echo "$T" | command grep -c .)" = 1 ] || { echo "$n trace files: [$T]" >> $L; break; }
    echo "$n trace $T $(stat -c %s $T) B $(sha256sum < $T | cut -c1-16)" >> $L
    $R/rcm-lb-dc $T > $D/rcm-64x4.out 2>&1; echo "$n rcm-lb-dc 64x4 rc=$?" >> $L
    $R/rcm-lb-dc $T --lb 1024x8 > $D/rcm-1024x8.out 2>&1; echo "$n rcm-lb-dc 1024x8 rc=$?" >> $L
    command grep -a '^v3\|^LB  *check\|^RC  *check' $D/rcm-1024x8.out >> $L
    local mm b44; mm=$(command grep -a -o 'mismatch_vs_LB0 [0-9]*' $D/rcm-1024x8.out | cut -d' ' -f2); b44=$(command grep -a -o 'bit44 [0-9]*' $D/rcm-1024x8.out | cut -d' ' -f2)
    if [ $n = gcc ]; then
      { [ "$mm" = 0 ] && [ "${b44:-0}" -gt 0 ] && command grep -a -q 'B5TL1-502.gcc_r=PASS' $D/grade.out && command grep -a -q '^WINDOW-OK' $D/window.out; } || { echo "STOP gcc gate: mismatch [$mm] bit44 [$b44] window [$(tail -1 $D/window.out)]" >> $L; break; }
      echo "gcc gate passed" >> $L
    fi
  done
  echo "$(date '+%F %T') end" >> $L; touch $R/CHAIN-DC-DONE
}
main "$@"; exit
