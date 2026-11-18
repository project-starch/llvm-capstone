#!/usr/bin/env bash
# Cases 02, 11 and 13 on PoisonCap's published artifact, built from the corpus sources of dev
# fe1d2dad2ebe, in the driver configuration of 2026-10-26 (runs driver-la and driver-la-rep2):
# 256 MiB heap from malloc, minimum request 16, lookaside on, C-library revocation off.
#   c19-driver-la, -rep2   the scored runs, timeout 300 s per case as on 2026-10-26
#   c19-long-la            not scored: case 11 from the same build with a 1,500 s timeout, in case
#                          it neither faults nor returns within 300 s
#   c19-diag-la            not scored: the diagnostic amalgamation (prints every free, flush and
#                          realloc), 1,500 s timeout
set -uo pipefail
P=$HOME/poisoncap-platform
K=$P/repo-19cases
cd "$P"
CFG=(--heap-bytes 268435456 --heap-min 16 --heap-malloc)
SEL=02,11,13
rm -f logs/c19-poisoncap.done

build() {  # $1 build name, $2 amalgamation
  rm -rf "builds/$1"
  bash "$K/capstone/ports/sqlite/poisoncap/platform/run-in-container.sh" \
    python3 "$K/capstone/ports/sqlite/poisoncap/corpus/build-cases.py" --platform "$P" \
    --amalgamation "$2" --ext-src sqlite/sqlite-src \
    --out "builds/$1" "${CFG[@]}" --only $SEL > "logs/build-$1.out" 2>&1
  echo "rc=$?" >> "logs/build-$1.out"
}
run() {  # $1 run name, $2 build, $3 port, $4 timeout
  rm -rf "runs/$1" "verdicts/$1"
  python3 "$K/capstone/ports/sqlite/poisoncap/corpus/run-cases.py" "runs/$1" --build "builds/$2" \
    --expect published --port "$3" --timeout "$4" --cases $SEL > "logs/run-$1.out" 2>&1
  echo "rc=$?" >> "logs/run-$1.out"
}
score() {  # $1 run name
  bash "$K/capstone/ports/sqlite/poisoncap/platform/run-in-container.sh" \
    python3 "$K/capstone/ports/sqlite/poisoncap/corpus/score-cases.py" --published "runs/$1" \
    --nm work/sdk/bin/llvm-nm --rootfs work/output/rootfs-riscv64-purecap --out "verdicts/$1" \
    > "logs/score-$1.out" 2>&1; echo "rc=$?" >> "logs/score-$1.out"
}

build c19-driver-la sqlite/amalgamation-published &
build c19-diag-la sqlite/amalgamation-published-diag &
wait
# One PoisonCap guest at a time: three guests started together failed their control gates
# (first attempt of this script, kept in logs/first-attempt-c19/).
run c19-driver-la c19-driver-la 10097 300; score c19-driver-la
run c19-driver-la-rep2 c19-driver-la 10097 300; score c19-driver-la-rep2
SEL=11
run c19-long-la c19-driver-la 10097 1500; score c19-long-la
SEL=02,11,13
run c19-diag-la c19-diag-la 10097 1500
echo done > logs/c19-poisoncap.done
