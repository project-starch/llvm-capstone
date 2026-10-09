#!/usr/bin/env bash
# Run the native builds with the arguments of gate.sh, in parallel, each plain and traced:
# results/<name>.txt (program output and MQ lines) with its exit status and /usr/bin/time -v.
#   usage: run-native.sh <out dir of build-native.sh> <results dir>
set -u
OUT=${1:?usage: run-native.sh <out> <results>}; R=${2:?usage: run-native.sh <out> <results>}
mkdir -p "$R"; cd "$OUT"
run() {
  local name=$1 in=$2; shift 2
  ( /usr/bin/time -v "$@" < "$in" > "$R/$name.txt" 2> "$R/$name.err"; echo "rc=$?" >> "$R/$name.err" ) &
}
run tracer-check-traced /dev/null ./tracer-check-traced
for v in "" -traced; do
  run glibc-simple$v /dev/null    ./glibc-simple$v
  run barnes$v       barnes.input ./barnes$v
  run espresso$v     /dev/null    ./espresso$v largest.espresso
  run cfrac$v        /dev/null    ./cfrac$v 17545186520507317056371138836327483792789528
  run mstress$v      /dev/null    ./mstress$v 1 50 25
  run sh6bench$v     /dev/null    env MQ_ADDR_SAMPLE=16 ./sh6bench$v 1
  run sh8bench$v     /dev/null    env MQ_ADDR_SAMPLE=16 ./sh8bench$v 1
  run mleak5$v       /dev/null    ./mleak$v 5
  run mleak50$v      /dev/null    ./mleak$v 50
done
wait
echo NATIVE-DONE > "$R/DONE"
