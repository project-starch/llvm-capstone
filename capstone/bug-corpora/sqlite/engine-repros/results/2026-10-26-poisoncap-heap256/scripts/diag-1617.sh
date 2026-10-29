#!/usr/bin/env bash
# DIAGNOSTIC ONLY: cases 16 and 17 on PoisonCap's diagnostic amalgamation (prints every free, flush and
# realloc), in the driver configuration (256 MiB from malloc) with minimum request 16 and with 64.
# Explains why the driver configuration misses 16 and 17; never a verdict.
set -uo pipefail
P=$HOME/poisoncap-platform
K=$P/repo-heap256
cd "$P"
one() {  # $1 name, $2 minimum request, $3 port
  rm -rf "builds/$1" "runs/$1"
  bash "$K/capstone/ports/sqlite/poisoncap/platform/run-in-container.sh" \
    python3 "$K/capstone/ports/sqlite/poisoncap/corpus/build-cases.py" --platform "$P" \
    --amalgamation sqlite/amalgamation-published-diag --ext-src sqlite/sqlite-src \
    --out "builds/$1" --heap-bytes 268435456 --heap-min "$2" --heap-malloc --only 16,17 > "logs/build-$1.out" 2>&1
  echo "rc=$?" >> "logs/build-$1.out"
  python3 "$K/capstone/ports/sqlite/poisoncap/corpus/run-cases.py" "runs/$1" --build "builds/$1" \
    --expect published --port "$3" --cases 16,17 > "logs/run-$1.out" 2>&1; echo "rc=$?" >> "logs/run-$1.out"
}
one diag-driver16-la 16 10099 &
one diag-driver64-la 64 10101 &
wait
echo done > logs/diag-1617.done
