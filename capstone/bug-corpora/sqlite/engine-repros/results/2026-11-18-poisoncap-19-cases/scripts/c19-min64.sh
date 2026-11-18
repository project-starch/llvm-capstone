#!/usr/bin/env bash
# DIAGNOSTIC ONLY, never a verdict: case 11 on PoisonCap's published amalgamation with minimum
# request 64 instead of 16 (256 MiB from malloc, lookaside on), the cross-check that
# 2026-10-26 used for cases 16 and 17: at 64 every block is 64-byte aligned, so no stale word has
# address bit 4 set. Runs after c19-long.sh, one guest at a time.
set -uo pipefail
P=$HOME/poisoncap-platform
K=$P/repo-19cases
cd "$P"
while [ ! -e logs/c19-long.done ]; do sleep 20; done
rm -rf builds/c19-min64-la runs/c19-min64-la verdicts/c19-min64-la
bash "$K/capstone/ports/sqlite/poisoncap/platform/run-in-container.sh" \
  python3 "$K/capstone/ports/sqlite/poisoncap/corpus/build-cases.py" --platform "$P" \
  --amalgamation sqlite/amalgamation-published --ext-src sqlite/sqlite-src \
  --out builds/c19-min64-la --heap-bytes 268435456 --heap-min 64 --heap-malloc --only 11 > logs/build-c19-min64-la.out 2>&1
echo "rc=$?" >> logs/build-c19-min64-la.out
python3 "$K/capstone/ports/sqlite/poisoncap/corpus/run-cases.py" runs/c19-min64-la --build builds/c19-min64-la \
  --expect published --port 10097 --timeout 300 --cases 11 > logs/run-c19-min64-la.out 2>&1
echo "rc=$?" >> logs/run-c19-min64-la.out
bash "$K/capstone/ports/sqlite/poisoncap/platform/run-in-container.sh" \
  python3 "$K/capstone/ports/sqlite/poisoncap/corpus/score-cases.py" --published runs/c19-min64-la \
  --nm work/sdk/bin/llvm-nm --rootfs work/output/rootfs-riscv64-purecap --out verdicts/c19-min64-la \
  > logs/score-c19-min64-la.out 2>&1; echo "rc=$?" >> logs/score-c19-min64-la.out
echo done > logs/c19-min64.done
