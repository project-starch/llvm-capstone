#!/bin/bash
# QEMU gate of the perlbench and x264 test stages (devmk, Temporal) on kit-etq3h18 (the bkit26-gt module's QEMU build,
# all2 params, exact-image vexec), two slots in parallel; their outputs are the board runs' twin.
main() {
  local S=${CAPSTONE_WORK:?}/spec B=${CAPSTONE_WORK:?}/wt-spec/capstone/ports/spec-cpu2017/board R=${CAPSTONE_WORK:?}/perf/rest
  unset CAPSTONE_MOVC_NULL_SCALAR CAPSTONE_R60; export CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock
  one() { CAPSTONE_QEMU_DOMAIN_LOCK=$HOME/.capstone-locks/prof-slot-$1.lock SPEC_CENSUS_MEM=936M taskset -c 16-31,48-63 nice -n 10 \
    python3 $B/run-stage.py run --backend qemu --kit $S/kit-etq3h18 --stage $2 --out $3 --tree $S/tree-boarddevmk/SPEC17 --step-s 60000 --twin-inband-max 67108864 > $3.log 2>&1; echo "$(date +%T) $(basename $3) rc=$?"; }
  one 1 $R/stages/500.perlbench_r-test-capstone-lifetimes-devmk $R/q-perl &
  one 2 $R/stages/525.x264_r-test-capstone-lifetimes-devmk $R/q-x264 &
  wait; echo QG-DONE
}
main "$@"; exit
