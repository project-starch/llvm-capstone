#!/usr/bin/env bash
# Two commands. The first shows the defect through the benchmark, the second
# reduces it to the one line that matters and shows it is not Capstone-specific.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source capstone/tests/capstone-test-env.sh

echo "== 1. the benchmark, as the nightly runs it =="
bash capstone/benchmarks/beebs/run-beebs-matmult-float.sh 2>&1 | grep -E "error:|__float128" | head -3 || true

echo "== 2. reduced, both targets =="
for t in capstone64-unknown-elf riscv64-unknown-elf; do
  err=$("$CAPSTONE_CLANG" -target "$t" -ffreestanding -c \
    capstone/tests/compiler-repros/H01-beebs-matmult-float-host-headers/src/float128.c \
    -o /dev/null 2>&1 | grep -m1 -oE 'error: .*' || true)
  printf '   %-24s %s\n' "$t" "${err:-accepted __float128}"
done
