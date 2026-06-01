#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-coremark.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
COREMARK_SRC_DIR=${COREMARK_SRC_DIR:-$CAPSTONE_TMP_ROOT/coremark-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/coremark-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj}
PORT_DIR="$SCRIPT_DIR/port"
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O2}
COREMARK_TOTAL_DATA_SIZE=${COREMARK_TOTAL_DATA_SIZE:-1200}
COREMARK_ITERATIONS=${COREMARK_ITERATIONS:-10}
OUT_DOM=${OUT_DOM:-$OUT_DIR/coremark_capstone.dom}

mkdir -p "$OUT_DIR" "$OBJ_DIR"

if [[ ! -f "$COREMARK_SRC_DIR/coremark.h" ]]; then
  echo "missing CoreMark source tree: $COREMARK_SRC_DIR" >&2
  exit 1
fi

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  "$DOMAIN_OPT_LEVEL"
  -I"$PORT_DIR"
  -I"$COREMARK_SRC_DIR"
  -DTOTAL_DATA_SIZE="$COREMARK_TOTAL_DATA_SIZE"
  -DITERATIONS="$COREMARK_ITERATIONS"
  -DPROFILE_RUN=1
  -Dmain=coremark_main
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -ffreestanding -O0 \
  -I"$PORT_DIR" \
  -I"$COREMARK_SRC_DIR" \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

for src in \
  "$COREMARK_SRC_DIR/core_main.c" \
  "$COREMARK_SRC_DIR/core_state.c" \
  "$SCRIPT_DIR/coremark_domain.c"
do
  obj="$OBJ_DIR/$(basename "${src%.c}").o"
  "$CLANG" "${COMMON_FLAGS[@]}" -c "$src" -o "$obj"
done

# Current benchmark-local runtime workarounds:
# - core_list_join.c at -O1: avoids the current high-opt list-path capability
#   copy trap while keeping the benchmark logic intact.
# - core_list_capstone.c: fixes upstream list sizing that still assumes 16-byte
#   per-node pointer storage and explicitly aligns the list storage for
#   capability-bearing list_head nodes on Capstone PureCap.
# - core_matrix_capstone.c: uses a local capability-safe matrix initializer while
#   the upstream function still triggers a mixed scalar/capability lowering bug.
# - core_util.c: avoid compiler-generated switch tables of capability-valued
#   addresses until generic static-cap table materialization exists.
# - core_portme.c: keep zero-initialized seed globals in .data so gp-relative
#   capability bounds still cover the full volatile seed set.
# - core_portme.c also keeps the canonical smoke on the list-only path for now
#   so later matrix-only faults do not hide list/runtime progress.
"$CLANG" "${COMMON_FLAGS[@]}" -fno-jump-tables -O1 \
  -Dcore_list_init=core_list_init_upstream_unused \
  -c "$COREMARK_SRC_DIR/core_list_join.c" \
  -o "$OBJ_DIR/core_list_join.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/core_list_capstone.c" \
  -o "$OBJ_DIR/core_list_capstone.o"

"$CLANG" "${COMMON_FLAGS[@]}" -Dcore_init_matrix=core_init_matrix_upstream_unused \
  -c "$COREMARK_SRC_DIR/core_matrix.c" \
  -o "$OBJ_DIR/core_matrix.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/core_matrix_capstone.c" \
  -o "$OBJ_DIR/core_matrix_capstone.o"

"$CLANG" "${COMMON_FLAGS[@]}" -fno-jump-tables \
  -c "$COREMARK_SRC_DIR/core_util.c" \
  -o "$OBJ_DIR/core_util.o"

"$CLANG" "${COMMON_FLAGS[@]}" -fno-zero-initialized-in-bss \
  -DCOREMARK_DEFAULT_EXECS=1 \
  -c "$SCRIPT_DIR/port/core_portme.c" \
  -o "$OBJ_DIR/core_portme.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/core_list_join.o" \
  "$OBJ_DIR/core_list_capstone.o" \
  "$OBJ_DIR/core_main.o" \
  "$OBJ_DIR/core_matrix.o" \
  "$OBJ_DIR/core_matrix_capstone.o" \
  "$OBJ_DIR/core_state.o" \
  "$OBJ_DIR/core_util.o" \
  "$OBJ_DIR/core_portme.o" \
  "$OBJ_DIR/coremark_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

echo "Built $OUT_DOM"

