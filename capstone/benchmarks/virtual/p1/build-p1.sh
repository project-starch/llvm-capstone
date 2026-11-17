#!/usr/bin/env bash
# build-p1.sh: the 103 microbenchmarks for P1's two SPEC arms (T1 = FRCDKMSspa3 SDK, T3 = FRCDKMSa3 SDK), with the
# recipe of wt-p1suite (origin/dev 21859e83 + lane p1-suite-native: -DBENCH_CONTEXT_COUNTERS via --context-counters).
main() {
  set -euo pipefail
  local H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}
  local P=$H/perf/p1suite V=$H/wt-p1suite/capstone/benchmarks/virtual A=$H/perf/cclever/arms
  local pin="taskset -c 16-31,48-63 nice -n 10 ionice -c3"
  export CAPSTONE_LLVM_BIN=$H/llvm-c9/bin PYTHONDONTWRITEBYTECODE=1 JOBS=12
  unset CAPSTONE_CLANG CCC_OVERRIDE_OPTIONS CAPSTONE_SDK CAPSTONE_COMPILE_LOG
  cd $P
  $pin python3 $V/heap-layout.py --sdk $A/FRCDKMSspa3/sdk --sdk $A/FRCDKMSa3/sdk --out $P/heap-layout.json
  rm -rf $P/build; mkdir -p $P/build
  $pin python3 $V/build.py --mode capstone --sdk $A/FRCDKMSspa3/sdk --heap-layout $P/heap-layout.json --context-counters --out $P/build/T1
  $pin python3 $V/build.py --mode capstone --sdk $A/FRCDKMSa3/sdk --heap-layout $P/heap-layout.json --context-counters --out $P/build/T3
  echo BUILD-P1-DONE
}
main "$@"; exit
