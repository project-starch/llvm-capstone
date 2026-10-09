#!/bin/sh
# Guest side: run the fixed-work C programs of mimalloc-bench (see build-domain.sh) under the virtual runtime, cheapest first,
# each with a time limit so that every run returns a verdict. Markers:
#   MB_BEGIN:<name>   MB_END:<name>:<exit status>   (137 = killed at the time limit)
# then MB_RUSAGE (mb-run: the launcher process's peak RSS, the counterpart of CheriBSD's
# time -l), the launcher's CAPSTONE_VM_SAMPLE lines (adapter counters and process memory every
# CAPSTONE_VM_SAMPLE_MS of launcher time) and final CAPSTONE_VM_STATS line, every MQ line of a
# traced build, and the program's own output (its oracle), all streamed live; at the end
# MB_OUT_LINES (all lines the run wrote, to check the stream is complete) and MB_OUT_SHA256
# over the program's own output.
# The stage may hold two optional files: "timeout" (seconds per program, default 1800) and
# "only" (names as in the markers, e.g. "cfrac cfrac-traced") to run a subset.
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_SAMPLE_MS=1000
T=1800; [ -f timeout ] && T=$(cat timeout)
MB_ONLY=; [ -f only ] && MB_ONLY=$(cat only)
echo MB_LIMIT:$T
echo MB_ONLY:$MB_ONLY
# run <name> <stdin file> <launcher arguments...>
# The stage disk is read-only, so the serial console is the only way out. Everything the run
# prints is streamed there as it happens (tail -f), so a long run shows its progress and a run
# that never ends still leaves every sample it took. The program's own output (everything but
# the result lines) is hashed at the end, to compare with the native run.
run() {
  name=$1; in=$2; shift 2
  case " ${MB_ONLY:-$name} " in *" $name "*) ;; *) return ;; esac
  echo MB_BEGIN:$name
  : > /tmp/$name.out
  ./mb-run $T ./capstone-vexec "$@" < $in > /tmp/$name.out 2>&1 &
  run_pid=$!
  tail -f /tmp/$name.out &
  tail_pid=$!
  wait $run_pid
  rc=$?
  # busybox tail polls once a second: let it reach the end of the file before stopping it.
  sleep 3
  kill $tail_pid; wait $tail_pid 2>/dev/null
  echo
  echo MB_END:$name:$rc
  grep -a -v -E '^(MB_RUSAGE|CAPSTONE_VM_STATS|CAPSTONE_VM_SAMPLE|MQ)' /tmp/$name.out > /tmp/$name.prog
  echo MB_OUT_LINES:$name:$(wc -l < /tmp/$name.out)
  echo MB_OUT_BYTES:$name:$(wc -c < /tmp/$name.prog)
  echo MB_OUT_SHA256:$name:$(sha256sum < /tmp/$name.prog | cut -d' ' -f1)
  rm -f /tmp/$name.out /tmp/$name.prog
}
# Plain programs first (footprint: CAPSTONE_VM_STATS), then the traced builds (MQ lines).
# The tracer's positive control first: it takes seconds.
run tracer-check-traced /dev/null tracer-check-traced.dom
for v in "" -traced; do
  run glibc-simple$v /dev/null    glibc-simple$v.dom
  run barnes$v       barnes.input barnes$v.dom
  run espresso$v     /dev/null    espresso$v.dom largest.espresso
  run cfrac$v        /dev/null    cfrac$v.dom 17545186520507317056371138836327483792789528
  run mstress$v      /dev/null    mstress$v.dom 1 50 25
  # sh6bench hands out tens of millions of distinct addresses: the tracer follows 1 in 16.
  export MQ_ADDR_SAMPLE=16
  run sh6bench$v     /dev/null    sh6bench$v.dom 1
  run sh8bench$v     /dev/null    sh8bench$v.dom 1
  unset MQ_ADDR_SAMPLE
  run mleak5$v       /dev/null    mleak$v.dom 5
  run mleak50$v      /dev/null    mleak$v.dom 50
done
rmmod capstone_vm
echo VIRTUAL_STAGED_DONE
