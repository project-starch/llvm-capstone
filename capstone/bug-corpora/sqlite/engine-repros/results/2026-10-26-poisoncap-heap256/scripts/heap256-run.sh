#!/usr/bin/env bash
# Run both arms of the 256 MiB corpus comparison twice each, then score the PoisonCap runs.
# PoisonCap: run-cases.py, single-user boot with runtime_revocation_default=0 (their kernel panics
# in a normal boot). Sublet: run-virtual-staged.py on the v6 platform, CPU as the v6 record.
set -uo pipefail
P=$HOME/poisoncap-platform
K=$P/repo-heap256
cd "$P"
export CAPSTONE_QEMU_CPU_EXTRA="svadu=false,x-capstone-supervisor-slots=32"
pc() {  # $1 run name, $2 port
  python3 "$K/capstone/ports/sqlite/poisoncap/corpus/run-cases.py" "runs/$1" --build builds/heap256-la \
    --expect published --port "$2" > "logs/run-$1.out" 2>&1; echo "rc=$?" >> "logs/run-$1.out"
  bash "$K/capstone/ports/sqlite/poisoncap/platform/run-in-container.sh" \
    python3 "$K/capstone/ports/sqlite/poisoncap/corpus/score-cases.py" --published "runs/$1" \
    --nm work/sdk/bin/llvm-nm --rootfs work/output/rootfs-riscv64-purecap --out "verdicts/$1" \
    > "logs/score-$1.out" 2>&1; echo "rc=$?" >> "logs/score-$1.out"
}
sl() {  # $1 run name
  python3 "$K/capstone/ports/sqlite/repro322/run-virtual-staged.py" --build builds/v6-sublet-heap256 \
    --platform platform-v6-heap256 --work "runs/$1" --llvm-bin "$HOME/spec-kit/toolchain/build/bin" \
    > "logs/run-$1.out" 2>&1; echo "rc=$?" >> "logs/run-$1.out"
}
(pc heap256-la 10093; pc heap256-la-rep2 10093) &
(sl v6-sublet-heap256; sl v6-sublet-heap256-rep2) &
wait
echo done > logs/heap256-run.done
