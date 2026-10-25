#!/usr/bin/env bash
# launch.sh memcached|mruby: the program's six boots (stock and protected, reps 1-3) in parallel,
# each its own VM, lock and fresh work directory under ~/sa-mem-mm/runs. Status lines in
# ~/sa-mem-mm/<program>-runs.status. MM_TIMEOUT sets each boot's timeout in seconds (default 3600).
set -u
. $HOME/sa-mem-mm/env.sh
prog=${1:?program}
ST=$S/$prog-runs.status; : > $ST
mkdir -p $S/runs
common=(--qemu $K/kit-sublet/qemu-system-riscv64 --images $K/kit-sublet/images --adapter $K/kit-sublet/adapter
        --run-staged $R/runtime/virtual/run-staged.py)
one() {   # arm rep
  local arm=$1 rep=$2 t=$(date +%s) img extra
  if [ $prog = memcached ]; then
    [ $arm = stock ] && img=$S/mc-build/domain/memcached.dom || img=$S/mc-build/sublet/memcached.dom
    extra=(--job $K/adapter-sublet/capstone-job --harness $K/mc/mc-harness-riscv-static --stats-client $S/mc-stats-riscv-static)
  else
    [ $arm = stock ] && img=$S/mruby/stock/src/mruby/build/capstone/bin/mruby || img=$S/mruby/prot/src/mruby/build/capstone/bin/mruby
    extra=(--driver $S/mstats.rb --ao $S/mruby/stock/src/mruby/benchmark/bm_ao_render.rb
           --smoke $R/ports/mruby/app/scripts/smoke.rb)
  fi
  CAPSTONE_QEMU_LOCK=$S/qemu-$prog-$arm-$rep.lock python3 $S/mm-run.py --program $prog --arm $arm --rep $rep \
    --timeout ${MM_TIMEOUT:-3600} --work $S/runs/$prog-$arm-$rep --image $img "${common[@]}" "${extra[@]}" > $S/runs/$prog-$arm-$rep.log 2>&1
  echo "RUN $prog $arm $rep rc=$? $(( $(date +%s) - t ))s" >> $ST
}
for rep in 1 2 3; do one stock $rep & one protected $rep & done
wait
echo "ALL-DONE" >> $ST
