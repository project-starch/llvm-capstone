#!/bin/sh
# Guest side: run the fixed-work C programs of mimalloc-bench (see build-domain.sh) under the virtual runtime, cheapest first,
# each with a time limit so that every run returns a verdict. Markers:
#   MB_BEGIN:<name>   MB_END:<name>:<exit status>   (137 = killed at the time limit)
# then MB_RUSAGE (mb-run: the launcher process's peak RSS, the counterpart of CheriBSD's
# time -l), the launcher's CAPSTONE_VM_STATS line, every MQ line of a traced build and the
# program's own output (its oracle).
# The stage may hold two optional files: "timeout" (seconds per program, default 1800) and
# "only" (names as in the markers, e.g. "cfrac cfrac-traced") to run a subset.
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1
T=1800; [ -f timeout ] && T=$(cat timeout)
MB_ONLY=; [ -f only ] && MB_ONLY=$(cat only)
echo MB_LIMIT:$T
echo MB_ONLY:$MB_ONLY
# run <name> <stdin file> <launcher arguments...>
run() {
  name=$1; in=$2; shift 2
  case " ${MB_ONLY:-$name} " in *" $name "*) ;; *) return ;; esac
  echo MB_BEGIN:$name
  ./mb-run $T ./capstone-vexec "$@" < $in > /tmp/$name.out 2>&1
  echo MB_END:$name:$?
  # The stage disk is read-only, so the serial console is the only way out: the result
  # lines in full, the rest of the output only up to 4000 bytes.
  grep -a -E '^(MB_RUSAGE|CAPSTONE_VM_STATS|MQ)' /tmp/$name.out
  grep -a -v -E '^(MB_RUSAGE|CAPSTONE_VM_STATS|MQ)' /tmp/$name.out | head -c 4000
  echo
  echo MB_OUT_BYTES:$name:$(wc -c < /tmp/$name.out)
}
# Plain programs first (footprint: CAPSTONE_VM_STATS), then the traced builds (MQ lines).
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
