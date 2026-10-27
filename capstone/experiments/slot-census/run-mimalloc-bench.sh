#!/bin/bash
# Slot census of mimalloc-bench's allocator benchmarks (C and C++), with the commands of its own bench.sh. These are
# allocator stress tests, not applications: they bound what an allocator must survive, and are reported apart from
# the application suites.
#
# usage: run-mimalloc-bench.sh OUT_DIR [TEST...]   (default: every test below)
# Clones mimalloc-bench at $COMMIT, builds bench/ with CMake (Release), and runs each test once in out/bench with
# procs=$PROCS (bench.sh's concurrency level). sh6bench and sh8bench need a separate download and are left out;
# the tests that drive external programs (gs, lua, lean, redis, rocksdb, z3, rbstress, spec) are left out too, as
# the application runners cover those programs. A cell is valid when the test exits 0 (the tests check nothing
# else).
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; shift
COMMIT=391382606681408e34d96ec54e9e6c79805aa3ac
PROCS=${SLOTCENSUS_MB_PROCS:-8}
. "$HERE/tracer.sh"
start_pass

MB=$TOOL/mimalloc-bench
git -c advice.detachedHead=false clone -q https://github.com/daanx/mimalloc-bench "$MB" &&
  git -C "$MB" -c advice.detachedHead=false checkout -q "$COMMIT" || exit 1
# sh6bench and sh8bench are built from a separate download (bench.sh fetches and patches it); without it their
# targets stop CMake, so the Linux branch that adds them is turned off
sed -i 's/^if(NOT APPLE)$/if(FALSE)  # slot census: sh6bench, sh8bench not downloaded/' "$MB/bench/CMakeLists.txt"
grep -q 'if(FALSE)  # slot census' "$MB/bench/CMakeLists.txt" \
  || { echo "run-mimalloc-bench: CMakeLists changed" >&2; exit 1; }
mkdir -p "$MB/out/bench"
(cd "$MB/out/bench" && cmake ../../bench -DCMAKE_BUILD_TYPE=Release > "$OUT/logs/build.log" 2>&1 &&
 make -j16 >> "$OUT/logs/build.log" 2>&1) || { echo "run-mimalloc-bench: build failed" >&2; exit 1; }
{
  provenance_tracer
  echo "mimalloc-bench $(git -C "$MB" log -1 --format='%H %cd')"
  echo "build cmake bench -DCMAKE_BUILD_TYPE=Release; procs=$PROCS"
} > "$OUT/provenance.txt"

# name | executable | arguments | stdin, as bench.sh runs them
TESTS=(
  "cfrac|cfrac|17545186520507317056371138836327483792789528|/dev/null"
  "espresso|espresso|../../bench/espresso/largest.espresso|/dev/null"
  "barnes|barnes||../../bench/barnes/input"
  "larsonN|larson|5 8 1000 5000 100 4141 $PROCS|/dev/null"
  "larsonN-sized|larson-sized|5 8 1000 5000 100 4141 $PROCS|/dev/null"
  "alloc-test1|alloc-test|1|/dev/null"
  "alloc-testN|alloc-test|$PROCS|/dev/null"
  "xmalloc-testN|xmalloc-test|-w $PROCS -t 5 -s 64|/dev/null"
  "cache-thrash1|cache-thrash|1 1000 1 2000000 $PROCS|/dev/null"
  "cache-thrashN|cache-thrash|$PROCS 1000 1 2000000 $PROCS|/dev/null"
  "cache-scratch1|cache-scratch|1 1000 1 2000000 $PROCS|/dev/null"
  "cache-scratchN|cache-scratch|$PROCS 1000 1 2000000 $PROCS|/dev/null"
  "malloc-large|malloc-large||/dev/null"
  "mstressN|mstress|$PROCS 50 25|/dev/null"
  "mleak10|mleak|5|/dev/null"
  "mleak100|mleak|50|/dev/null"
  "rptestN|rptest|$PROCS 0 1 2 500 1000 100 8 16000|/dev/null"
  "glibc-simple|glibc-simple||/dev/null"
  "glibc-thread|glibc-thread|$PROCS|/dev/null"
)
WANT=("$@")
: > "$OUT/cells.tsv"
cd "$MB/out/bench" || exit 1
for t in "${TESTS[@]}"; do
  IFS='|' read -r name exe args input <<< "$t"
  if [ ${#WANT[@]} -gt 0 ] && [[ " ${WANT[*]} " != *" $name "* ]]; then continue; fi
  cell=mb-$name
  read -r -a argv <<< "$args"
  s=$(date +%s)
  measured "$OUT/$cell.jsonl" "./$exe" "${argv[@]}" < "$input" \
    > "$OUT/logs/$cell.log" 2>&1
  rc=$?
  v=INVALID; [ $rc = 0 ] && v=valid
  record_cell "$cell" "$exe" "$v" "rc=$rc $(( $(date +%s) - s ))s"
done
echo CENSUS_DONE >> "$OUT/cells.tsv"
