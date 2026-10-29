#!/usr/bin/env bash
# The block port on the corpus, in PoisonCap's driver configuration (256 MiB heap from malloc,
# minimum request 16, lookaside on), next to the PoisonCap runs driver-la and driver-la-rep2.
#   block       sublet-3220000-memsys5-block.patch: block bounds, stock rounding and geometry,
#               every successful realloc a new lifetime. Twice.
#   block-keeps the ablation (-DSUBLET_REALLOC_KEEPS_LIFETIME): a realloc that keeps its block
#               returns the old lifetime. Twice.
# Each build carries the two standing controls and the four realloc and capacity controls.
set -uo pipefail
P=$HOME/poisoncap-platform
K=$P/repo-heap256
CTL=$K/capstone/bug-corpora/sqlite/engine-repros/controls
cd "$P"
export CAPSTONE_QEMU_CPU_EXTRA="svadu=false,x-capstone-supervisor-slots=32"
export CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock
mkdir -p "$(dirname "$CAPSTONE_QEMU_LOCK")"
CFG=(--heap-bytes 268435456 --heap-min 16 --heap-malloc)
OBS=()
for c in control_uaf_mem5 control_bounds_mem5 control_realloc_same_mem5 control_realloc_shrink_mem5 \
         control_realloc_fail_mem5 control_capacity_mem5; do OBS+=(--observe "$CTL/$c.c"); done

build() {  # $1 build name, then extra flags
  local name=$1; shift
  rm -rf "builds/$name"
  python3 "$K/capstone/ports/sqlite/repro322/build-virtual.py" --sdk "$P/sdk-v6" \
    --amalgamation "$HOME/pg-virtual-kit/inputs/sqlite-amalg" --ext-src "$HOME/pg-virtual-kit/inputs/sqlite-src-3220000" \
    --sublet --block-port "$@" "${CFG[@]}" "${OBS[@]}" --out "builds/$name" > "logs/build-$name.out" 2>&1
  echo "rc=$?" >> "logs/build-$name.out"
}
sl() {  # $1 run name, $2 build, $3 domain lock
  rm -rf "runs/$1"
  CAPSTONE_QEMU_DOMAIN_LOCK=$HOME/.capstone-locks/$3.lock \
  python3 "$K/capstone/ports/sqlite/repro322/run-virtual-staged.py" --build "builds/$2" \
    --platform platform-v6-heap256 --work "runs/$1" --llvm-bin "$HOME/spec-kit/toolchain/build/bin" \
    > "logs/run-$1.out" 2>&1; echo "rc=$?" >> "logs/run-$1.out"
}
build v6-sublet-block &
build v6-sublet-block-keeps --realloc-keeps-lifetime &
wait
(sl v6-sublet-block v6-sublet-block block-a; sl v6-sublet-block-rep2 v6-sublet-block block-a) &
(sl v6-sublet-block-keeps v6-sublet-block-keeps block-b; sl v6-sublet-block-keeps-rep2 v6-sublet-block-keeps block-b) &
wait
echo done > logs/block-port.done
