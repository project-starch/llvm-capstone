#!/usr/bin/env bash
# The main row of the corpus comparison, configured as PoisonCap's own speedtest1 main() configures
# memsys5: a 256 MiB heap from malloc, minimum request 16, on both arms. Lookaside on (SQLite's default).
#   heap256-driver.sh smoke   build cases 00 and 03 on both arms and run them once
#   heap256-driver.sh full    build every case on both arms, run each arm twice, score PoisonCap
set -uo pipefail
MODE=${1:?smoke or full}
P=$HOME/poisoncap-platform
K=$P/repo-heap256
cd "$P"
export CAPSTONE_QEMU_CPU_EXTRA="svadu=false,x-capstone-supervisor-slots=32"
export CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock
mkdir -p "$(dirname "$CAPSTONE_QEMU_LOCK")"
CFG=(--heap-bytes 268435456 --heap-min 16 --heap-malloc)
if [ "$MODE" = smoke ]; then ONLY=(--only 00,03); CASES=(--cases 00,03); TAG=smoke-; else ONLY=(); CASES=(); TAG=; fi
PB=builds/${TAG}driver-la
SB=builds/${TAG}v6-sublet-driver
rm -rf "$PB" "$SB"

(bash "$K/capstone/ports/sqlite/poisoncap/platform/run-in-container.sh" \
   python3 "$K/capstone/ports/sqlite/poisoncap/corpus/build-cases.py" --platform "$P" \
   --amalgamation sqlite/amalgamation-published --ext-src sqlite/sqlite-src \
   --out "$PB" "${CFG[@]}" "${ONLY[@]}" > "logs/build-${TAG}driver-la.out" 2>&1
 echo "rc=$?" >> "logs/build-${TAG}driver-la.out") &
(python3 "$K/capstone/ports/sqlite/repro322/build-virtual.py" --sdk "$P/sdk-v6" \
   --amalgamation "$HOME/pg-virtual-kit/inputs/sqlite-amalg" --ext-src "$HOME/pg-virtual-kit/inputs/sqlite-src-3220000" \
   --sublet "${CFG[@]}" "${ONLY[@]}" \
   --observe "$K/capstone/bug-corpora/sqlite/engine-repros/controls/control_uaf_mem5.c" \
   --observe "$K/capstone/bug-corpora/sqlite/engine-repros/controls/control_bounds_mem5.c" \
   --out "$SB" > "logs/build-${TAG}v6-sublet-driver.out" 2>&1
 echo "rc=$?" >> "logs/build-${TAG}v6-sublet-driver.out") &
wait

pc() {  # $1 run name
  rm -rf "runs/$1" "verdicts/$1"
  python3 "$K/capstone/ports/sqlite/poisoncap/corpus/run-cases.py" "runs/$1" --build "$PB" \
    --expect published --port 10097 "${CASES[@]}" > "logs/run-$1.out" 2>&1; echo "rc=$?" >> "logs/run-$1.out"
  bash "$K/capstone/ports/sqlite/poisoncap/platform/run-in-container.sh" \
    python3 "$K/capstone/ports/sqlite/poisoncap/corpus/score-cases.py" --published "runs/$1" \
    --nm work/sdk/bin/llvm-nm --rootfs work/output/rootfs-riscv64-purecap --out "verdicts/$1" \
    > "logs/score-$1.out" 2>&1; echo "rc=$?" >> "logs/score-$1.out"
}
sl() {  # $1 run name
  rm -rf "runs/$1"
  CAPSTONE_QEMU_DOMAIN_LOCK=$HOME/.capstone-locks/heap256-driver.lock \
  python3 "$K/capstone/ports/sqlite/repro322/run-virtual-staged.py" --build "$SB" \
    --platform platform-v6-heap256 --work "runs/$1" --llvm-bin "$HOME/spec-kit/toolchain/build/bin" "${CASES[@]}" \
    > "logs/run-$1.out" 2>&1; echo "rc=$?" >> "logs/run-$1.out"
}
if [ "$MODE" = smoke ]; then
  (pc smoke-driver-la) &
  (sl smoke-v6-sublet-driver) &
else
  (pc driver-la; pc driver-la-rep2) &
  (sl v6-sublet-driver; sl v6-sublet-driver-rep2) &
fi
wait
echo done > "logs/heap256-driver-$MODE.done"
