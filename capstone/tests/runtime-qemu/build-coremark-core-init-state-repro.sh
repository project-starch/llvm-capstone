#!/usr/bin/env bash
set -euo pipefail

# Build a minimal CoreMark-based domain that only calls core_init_state().
# Uses core_state_capstone.c (flat 2D pattern arrays) instead of the upstream
# pointer-table version so this probe validates the fix rather than reproducing
# the old helper_cscincoffset/rs1_v->tag failure.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

REPO_ROOT=${CAPSTONE_REPO_ROOT}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
COREMARK_SRC_DIR=${COREMARK_SRC_DIR:-$TMP_ROOT/coremark-src}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}
OBJ_DIR=${OBJ_DIR:-$TMP_ROOT/coremark-core-init-state-repro-build/obj}
REPRO_DIR="$SCRIPT_DIR/coremark-core-init-state-repro"
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
OUT_DOM="$OUT_DIR/coremark_core_init_state_repro.dom"

mkdir -p "$TMP_ROOT" "$OUT_DIR" "$OBJ_DIR"

bash "$REPO_ROOT/capstone/benchmarks/coremark/fetch-coremark.sh" >/dev/null

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature -Xclang +m
  -ffreestanding
  -fno-builtin
  -O2
  -I"$REPO_ROOT/capstone/benchmarks/coremark/port"
  -I"$COREMARK_SRC_DIR"
  -DTOTAL_DATA_SIZE=1200
  -DITERATIONS=10
  -DPROFILE_RUN=1
)

"$CLANG" \
  -target capstone64-unknown-elf \
  -Xclang -target-feature -Xclang +m \
  -ffreestanding \
  -O0 \
  -I"$REPO_ROOT/capstone/benchmarks/coremark/port" \
  -I"$COREMARK_SRC_DIR" \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

# Compile upstream core_state.c with core_init_state renamed so the local
# override in core_state_capstone.c wins at link time.
"$CLANG" "${COMMON_FLAGS[@]}" \
  -Dcore_init_state=core_init_state_upstream_unused \
  -c "$COREMARK_SRC_DIR/core_state.c" \
  -o "$OBJ_DIR/core_state.o"

# Local core_init_state: flat 2D pattern arrays, no capability table loads.
"$CLANG" "${COMMON_FLAGS[@]}" \
  -fno-jump-tables \
  -c "$REPO_ROOT/capstone/benchmarks/coremark/core_state_capstone.c" \
  -o "$OBJ_DIR/core_state_capstone.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -fno-jump-tables -O0 \
  -c "$COREMARK_SRC_DIR/core_util.c" \
  -o "$OBJ_DIR/core_util.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -fno-zero-initialized-in-bss \
  -DCOREMARK_DEFAULT_EXECS=1 \
  -c "$REPO_ROOT/capstone/benchmarks/coremark/port/core_portme.c" \
  -o "$OBJ_DIR/core_portme.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$REPRO_DIR/domain.c" \
  -o "$OBJ_DIR/domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/core_state.o" \
  "$OBJ_DIR/core_state_capstone.o" \
  "$OBJ_DIR/core_util.o" \
  "$OBJ_DIR/core_portme.o" \
  "$OBJ_DIR/domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

echo "Built $OUT_DOM"

