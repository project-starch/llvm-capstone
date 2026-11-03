#!/bin/bash
# PHP's own benchmark scripts on the virtual profile against plain RISC-V, end to end, on the
# profiling host (p12, the kit and census platform under ~/prof-dev):
#
#   1. prepare PHP 5.0.0 (../app/prepare-sources.sh), build both arms (build-arms.sh)
#   2. adapt php-src 8.3.17's Zend/bench.php and Zend/micro_bench.php (adapt-scripts.py)
#   3. profile, all at once (profile-php.sh): the 18 per-test scripts (one VM per arm, one window per
#      test), bench-reduced.php and micro_bench.php in both arms, and the full bench.php in the plain
#      arm only (the Capstone arm does not finish it in an hour)
#   4. prof-compare.py plain -> capstone and prof-buckets.py for each pair; pertest-table.py
#
# usage: run-all.sh WORK PHP_SRC_SCRIPTS [MICRO_N]
#   WORK             new directory below which everything goes (it is also the guest's share)
#   PHP_SRC_SCRIPTS  a directory with php-src 8.3.17's Zend/bench.php and Zend/micro_bench.php
#   MICRO_N          micro_bench.php's N (default 100000: a fiftieth of the script's 5000000, so that the
#                    Capstone arm stays inside the one-hour limit; reported as REDUCED)
# env: PHP_500_CACHE_ROOT (a directory holding php-5.0.0.tar.gz, else it is fetched), LIMIT (3600)
#      and everything build-arms.sh and profile-php.sh read.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
WORK=${1:?usage: run-all.sh WORK PHP_SRC_SCRIPTS [MICRO_N]}; SRC=$(cd "${2:?}" && pwd -P); MICRO_N=${3:-100000}
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd -P)
KIT=${KIT:-$HOME/prof-dev/kit3}
TREE=$(bash "$HERE/../app/prepare-sources.sh" "$WORK/php-src" | tail -n 1)
bash "$HERE/build-arms.sh" "$TREE" "$WORK/arms"
python3 -I "$HERE/adapt-scripts.py" "$SRC" "$WORK/scripts" "$MICRO_N"
mkdir -p "$WORK/runs"
pids=()
prof() { local arm=$1 name=$2; shift 2
          SHARE=$WORK bash "$HERE/profile-php.sh" "$arm" "$WORK/arms/$arm/php" "$WORK/runs/$name-$arm" "$@" \
            > "$WORK/runs/$name-$arm.log" 2>&1 & pids+=($!); }
for a in capstone plain; do
  prof "$a" pertest "$WORK"/scripts/pertest/*.php
  for s in bench-reduced micro_bench; do prof "$a" "$s" "$WORK/scripts/$s.php"; done
done
prof plain bench "$WORK/scripts/bench.php"
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
for s in pertest bench-reduced micro_bench; do
  python3 "$KIT/tools/prof-compare.py" "$WORK/runs/$s-plain" "$WORK/runs/$s-capstone" --top 15 \
    --json "$WORK/runs/$s-compare.json" > "$WORK/runs/$s-compare.txt"
  python3 "$KIT/tools/prof-buckets.py" "$WORK/runs/$s-plain" "$WORK/runs/$s-capstone" > "$WORK/runs/$s-buckets.txt"
  for a in plain capstone; do echo "== $s $a: $(tr -d '\n ' < "$WORK/runs/$s-$a/result.json")"; done
  head -n 20 "$WORK/runs/$s-compare.txt"
done
python3 -I "$HERE/pertest-table.py" "$WORK/runs/pertest-plain" "$WORK/runs/pertest-capstone" | tee "$WORK/runs/pertest-table.txt"
exit $rc
