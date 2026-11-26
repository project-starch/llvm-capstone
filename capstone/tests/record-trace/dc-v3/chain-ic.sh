#!/bin/bash
# chain-ic.sh (nested lane, 2026-11-27): the mcf v3 trace under QEMU -icount, at shift 7 (128 ns per instruction) and
# then shift 6 (64 ns). Amendment 1: the trace kernel ticks at 250 Hz and g1b's at 100 Hz (~228 ns per instruction), so
# silicon's ticks per instruction fall at ~91 ns here, between shift 6 and 7 (shift 8, the first plan, overshoots 2.8x). Under -icount the
# aclint mtime follows QEMU_CLOCK_VIRTUAL (hw/intc/riscv_aclint.c:44, :392), so the timer's M round trips per instruction
# are set by the shift and not by the host's speed or the plugin's slowdown. Each run is read as chain-dc.sh reads mcf
# (dc-window.py, grade-spec-chain-v2.py, rcm-lb-dc --lb 1024x8, whose LB0 is the 64x4) and its flush counts are logged
# from the trace's stats. Shift 6 runs only if shift 7 passes the gate: PASS, WINDOW-OK, v3 mismatch_vs_LB0 0, bit44 > 0.
# Writes chain-ic.log and CHAIN-IC-DONE.
# Dry run: CIC_DRY=1 CIC_ROOT=<dir> replaces the trace with runs/spec-mcf's recorded artifacts and rcm with its recorded
# rcm-1024x8.out; CIC_DRY_FAIL=1 also replaces the console with B3's (no window), so the gate's stop branch runs.
main() {
  local R=<hdd>/capstone/nested/rectrace-dc X=<hdd>/capstone/b5b2linux BS=<hdd>/capstone/c4/b5b2spec/board-stages
  local O=${CIC_ROOT:-$R}
  cd $R || return 2; local L=$O/chain-ic.log
  for s in ${CIC_SHIFTS:-7 6}; do
    local n=mcf-ic$s D=$O/runs/spec-mcf-ic$s
    echo "$(date '+%F %T') begin $n (-icount shift=$s)" >> $L
    if [ -n "${CIC_DRY:-}" ]; then
      mkdir -p $D $O/traces; local S=$R/runs/spec-mcf
      cp $S/boot.log $S/snap.txt $S/dcplugin.txt $D/; [ -z "${CIC_DRY_FAIL:-}" ] || cp $R/diag/B3/boot.log $D/boot.log
      ln -sf $R/traces/spec-mcf.1497224 $O/traces/spec-$n.dry; cp $R/traces/spec-mcf.1497224.stats $O/traces/spec-$n.dry.stats
      echo "$(date '+%F %T') $n rc=DRY" >> $L
    else
      TD_ICOUNT=shift=$s bash $R/trace-spec-dc-ic.sh $n $BS/B5TL1/505.mcf_r >> $L 2>&1; echo "$(date '+%F %T') $n rc=$?" >> $L
    fi
    python3 $R/dc-window.py $D/boot.log $D/snap.txt $D/dcplugin.txt > $D/window.out 2>&1; echo "$n window rc=$? $(tail -1 $D/window.out)" >> $L
    SPEC_STAGE_ROOT=$BS python3 $X/grade-spec-chain-v2.py $D/boot.log $D/grade B5TL1-505.mcf_r > $D/grade.out 2>&1; echo "$n grade rc=$? $(tail -1 $D/grade.out)" >> $L
    local T; T=$(ls $O/traces/spec-$n.* 2>/dev/null | command grep -av stats)
    [ "$(echo "$T" | command grep -c .)" = 1 ] || { echo "$n trace files: [$T]" >> $L; break; }
    echo "$n trace $T $(stat -L -c %s $T) B" >> $L
    command grep -a -E '^(instructions_all|instructions_protected|edges_lb_flush|edges_trap|edges_root|edges_sret|edges_mret) ' $T.stats | tr '\n' ' ' | sed "s/^/$n stats /" >> $L; echo >> $L
    if [ -n "${CIC_DRY:-}" ]; then cp $R/runs/spec-mcf/rcm-1024x8.out $D/rcm-1024x8.out; echo "$n rcm-lb-dc 1024x8 rc=DRY" >> $L
    else taskset -c 16-27 $R/rcm-lb-dc $T --lb 1024x8 > $D/rcm-1024x8.out 2>&1; echo "$n rcm-lb-dc 1024x8 rc=$?" >> $L; fi
    command grep -a '^v3\|^LB  *check\|^RC  *check' $D/rcm-1024x8.out >> $L
    local mm b44; mm=$(command grep -a -o 'mismatch_vs_LB0 [0-9]*' $D/rcm-1024x8.out | cut -d' ' -f2); b44=$(command grep -a -o 'bit44 [0-9]*' $D/rcm-1024x8.out | cut -d' ' -f2)
    { [ "$mm" = 0 ] && [ "${b44:-0}" -gt 0 ] && command grep -a -q 'B5TL1-505.mcf_r=PASS' $D/grade.out && command grep -a -q '^WINDOW-OK' $D/window.out; } || { echo "STOP $n gate: mismatch [$mm] bit44 [$b44] window [$(tail -1 $D/window.out)]" >> $L; break; }
    echo "$n gate passed" >> $L
  done
  echo "$(date '+%F %T') end" >> $L; touch $O/CHAIN-IC-DONE
}
main "$@"; exit
