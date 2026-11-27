#!/bin/bash
# chain-tk.sh (nested lane, 2026-11-27): mcf's v3 trace at -icount shift=6 again, now with the tracer's TICKS side
# file (trace-spec-dc-tk.sh, qemu/build-tk), so each inter-tick interval carries its exact instruction count and the D$
# model's misses. It is read as chain-ic.sh read mcf-ic6 (dc-window.py, grade-spec-chain-v2.py, rcm-lb-dc --lb
# 1024x8), plus rcm-lb-dc-ticks --ticks for the per-tick replay rows and tick-bins.py for the join.
# Gate, as chain-ic.sh's plus two: PASS, WINDOW-OK, v3 mismatch_vs_LB0 0, bit44 > 0; the ticks file is non-empty; and
# rcm-lb-dc-ticks agrees with rcm-lb-dc on removed_lb (the added rows change nothing). Writes chain-tk.log, CHAIN-TK-DONE.
# Dry run: CTK_DRY=1 CTK_ROOT=<dir> uses runs/spec-mcf-ic6's recorded artifacts, an empty replay-ticks file and the
# tk-smoke run's ticks file, so the gate passes and tick-bins.py fails (no rows); CTK_DRY_OK=1 skips tick-bins.py;
# CTK_DRY_FAIL=1 also replaces the console with B3's (no window), so the gate's stop branch runs.
main() {
  local R=<hdd>/capstone/nested/rectrace-dc X=<hdd>/capstone/b5b2linux BS=<hdd>/capstone/c4/b5b2spec/board-stages
  local O=${CTK_ROOT:-$R}
  cd $R || return 2
  local L=$O/chain-tk.log n=mcf-tk6
  local D=$O/runs/spec-$n
  echo "$(date '+%F %T') begin $n (-icount shift=6, ticks side file)" >> $L
  if [ -n "${CTK_DRY:-}" ]; then
    mkdir -p $D $O/traces; local S=$R/runs/spec-mcf-ic6
    cp $S/boot.log $S/snap.txt $S/dcplugin.txt $D/; [ -z "${CTK_DRY_FAIL:-}" ] || cp $R/diag/B3/boot.log $D/boot.log
    ln -sf $R/traces/spec-mcf-ic6.2017382 $O/traces/spec-$n.dry; cp $R/traces/spec-mcf-ic6.2017382.stats $O/traces/spec-$n.dry.stats
    cp $(ls $R/traces/spec-tk-smoke.*.ticks | head -1) $O/traces/spec-$n.dry.ticks
    echo "$(date '+%F %T') $n rc=DRY" >> $L
  else
    TD_ICOUNT=shift=6 bash $R/trace-spec-dc-tk.sh $n $BS/B5TL1/505.mcf_r >> $L 2>&1; echo "$(date '+%F %T') $n rc=$?" >> $L
  fi
  python3 $R/dc-window.py $D/boot.log $D/snap.txt $D/dcplugin.txt > $D/window.out 2>&1; echo "$n window rc=$? $(tail -1 $D/window.out)" >> $L
  SPEC_STAGE_ROOT=$BS python3 $X/grade-spec-chain-v2.py $D/boot.log $D/grade B5TL1-505.mcf_r > $D/grade.out 2>&1; echo "$n grade rc=$? $(tail -1 $D/grade.out)" >> $L
  local T; T=$(ls $O/traces/spec-$n.* 2>/dev/null | command grep -av -e '\.stats$' -e '\.ticks$')
  [ "$(echo "$T" | command grep -c .)" = 1 ] || { echo "STOP $n trace files: [$T]" >> $L; return 1; }
  echo "$n trace $T $(stat -L -c %s $T) B ticks $(wc -l < $T.ticks) rows" >> $L
  command grep -a -E '^(instructions_all|instructions_protected|edges_lb_flush|edges_trap|edges_root|edges_sret|edges_mret) ' $T.stats | tr '\n' ' ' | sed "s/^/$n stats /" >> $L; echo >> $L
  if [ -n "${CTK_DRY:-}" ]; then
    cp $R/runs/spec-mcf-ic6/rcm-1024x8.out $D/rcm-1024x8.out; cp $R/runs/spec-mcf-ic6/rcm-1024x8.out $D/rcm-ticks.out
    : > $D/replay-ticks.txt; echo "$n rcm rc=DRY" >> $L
  else
    taskset -c 16 $R/rcm-lb-dc $T --lb 1024x8 > $D/rcm-1024x8.out 2>&1 & local p1=$!
    taskset -c 17 $R/rcm-lb-dc-ticks $T --lb 1024x8 --ticks $D/replay-ticks.txt > $D/rcm-ticks.out 2>&1 & local p2=$!
    wait $p1; local r1=$?; wait $p2; local r2=$?
    echo "$n rcm-lb-dc 1024x8 rc=$r1 rcm-lb-dc-ticks rc=$r2" >> $L
  fi
  command grep -a '^v3\|^LB  *check\|^RC  *check\|^thin' $D/rcm-1024x8.out $D/rcm-ticks.out >> $L
  local mm b44 ra rb; mm=$(command grep -a -o 'mismatch_vs_LB0 [0-9]*' $D/rcm-1024x8.out | cut -d' ' -f2); b44=$(command grep -a -o 'bit44 [0-9]*' $D/rcm-1024x8.out | cut -d' ' -f2)
  ra=$(command grep -a 'pop removed_lb' $D/rcm-1024x8.out | awk '{print $5}'); rb=$(command grep -a 'pop removed_lb' $D/rcm-ticks.out | awk '{print $5}')
  { [ "$mm" = 0 ] && [ "${b44:-0}" -gt 0 ] && [ -n "$ra" ] && [ "$ra" = "$rb" ] && [ -s $T.ticks ] \
      && command grep -a -q 'B5TL1-505.mcf_r=PASS' $D/grade.out && command grep -a -q '^WINDOW-OK' $D/window.out; } \
    || { echo "STOP $n gate: mismatch [$mm] bit44 [$b44] removed [$ra]/[$rb] window [$(tail -1 $D/window.out)]" >> $L; echo "$(date '+%F %T') end" >> $L; touch $O/CHAIN-TK-DONE; return 1; }
  echo "$n gate passed" >> $L
  if [ -n "${CTK_DRY_OK:-}" ]; then echo "$n bins rc=DRY" >> $L
  else python3 $R/tick-bins.py $D/replay-ticks.txt $T.ticks > $D/bins.out 2>&1; echo "$n bins rc=$? $(head -1 $D/bins.out)" >> $L; fi
  echo "$(date '+%F %T') end" >> $L; touch $O/CHAIN-TK-DONE
}
main "$@"; exit
