#!/bin/sh
# Cleaner temporal-safety overhead: calibration-subtracted, averaged.
# CAL = empty lua (startup+linking, no workload); RUN = full benchmark.
# workload instrs = mean(RUN) - mean(CAL), per config.
CFG="${1:-spatial}"
case "$CFG" in
  spatial)  REV=0; EVERY=0; REPS=3 ;;
  temporal) REV=1; EVERY=0; REPS=3 ;;
  eager)    REV=1; EVERY=1; REPS=1 ;;   # eager is O(100x): one trial
esac
sysctl security.cheri.runtime_revocation_default=$REV >/dev/null 2>&1
sysctl security.cheri.runtime_revocation_every_free_default=$EVERY >/dev/null 2>&1
cd /root/lua-bench || exit 1
echo "==CFG $CFG rev=$REV every=$EVERY reps=$REPS=="
i=0; while [ $i -lt $REPS ]; do i=$((i+1)); printf 'CAL '; ./runbench "./lua -e 'os.exit(0)'"; done
i=0; while [ $i -lt $REPS ]; do i=$((i+1)); printf 'RUN '; ./runbench "./lua binary-trees.lua 6"; done
echo "===CHERI-BASELINE-END cfg=$CFG==="
