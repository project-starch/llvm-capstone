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
  "$COREMARK_SRC_DIR/core_list_join.c" \
  "$COREMARK_SRC_DIR/core_main.c" \
  "$COREMARK_SRC_DIR/core_matrix.c" \
  "$COREMARK_SRC_DIR/core_state.c" \
  "$COREMARK_SRC_DIR/core_util.c" \
  "$SCRIPT_DIR/port/core_portme.c" \
  "$SCRIPT_DIR/coremark_domain.c"
do
  obj="$OBJ_DIR/$(basename "${src%.c}").o"
  "$CLANG" "${COMMON_FLAGS[@]}" -c "$src" -o "$obj"
done

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/core_list_join.o" \
  "$OBJ_DIR/core_main.o" \
  "$OBJ_DIR/core_matrix.o" \
  "$OBJ_DIR/core_state.o" \
  "$OBJ_DIR/core_util.o" \
  "$OBJ_DIR/core_portme.o" \
  "$OBJ_DIR/coremark_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

echo "Built $OUT_DOM"

