#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RV8_BENCH=sha512

source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
"$SCRIPT_DIR/fetch-rv8.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
RV8_SRC_DIR=${RV8_SRC_DIR:-$CAPSTONE_TMP_ROOT/rv8-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/rv8-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-$RV8_BENCH}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/rv8_${RV8_BENCH}_capstone.dom}

BEEBS_DIR=$REPO_ROOT/capstone/benchmarks/beebs
ADAPTED_DIR=$SCRIPT_DIR/adapted
STRING_SRC=$BEEBS_DIR/adapted/beebs_freestanding_string.c
DOMAIN_SRC=$BEEBS_DIR/beebs_simple_domain.c
SRC=$RV8_SRC_DIR/src/sha512.c
PATCHED_SRC=$OUT_DIR/sha512_capstone.c

for f in "$SRC" "$STRING_SRC" "$DOMAIN_SRC" "$ADAPTED_DIR/rv8_capstone_preamble.h"; do
  [[ -f "$f" ]] || { echo "missing required source: $f" >&2; exit 1; }
done

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Strip hosted includes (preamble + string lib replace them); keep <stdint.h>
# (freestanding, compiler-provided -- needed for uintN_t). Drop the upstream
# main() (its hash loop + printf; our tail drives a reduced, checked run).
sed -E -e '/^[[:space:]]*#[[:space:]]*include[[:space:]]+<(stdio|stdlib|string|assert)\.h>/d' \
       -e '/^int main\(\)/,$d' \
       "$SRC" > "$PATCHED_SRC"

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  "$DOMAIN_OPT_LEVEL"
  -include "$ADAPTED_DIR/rv8_capstone_preamble.h"
  -I"$ADAPTED_DIR"
  -Wno-incompatible-library-redeclaration
  -Wno-implicit-function-declaration
  -Wno-builtin-requires-header
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$START_SRC" -o "$OBJ_DIR/start.o"

objs=("$OBJ_DIR/start.o")
"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_SRC" -o "$OBJ_DIR/sha512.o"; objs+=("$OBJ_DIR/sha512.o")
"$CLANG" "${COMMON_FLAGS[@]}" -c "$ADAPTED_DIR/rv8_sha512_tail.c" -o "$OBJ_DIR/tail.o"; objs+=("$OBJ_DIR/tail.o")
"$CLANG" "${COMMON_FLAGS[@]}" -DCAPSTONE_HEAP_SIZE=262144 -c "$ADAPTED_DIR/cap_heap.c" -o "$OBJ_DIR/malloc.o"; objs+=("$OBJ_DIR/malloc.o")
"$CLANG" "${COMMON_FLAGS[@]}" -DCAPSTONE_HEAP_SIZE=262144 -c "$ADAPTED_DIR/umm/umm_malloc.c" -o "$OBJ_DIR/umm.o"; objs+=("$OBJ_DIR/umm.o")
"$CLANG" "${COMMON_FLAGS[@]}" -c "$STRING_SRC" -o "$OBJ_DIR/string.o"; objs+=("$OBJ_DIR/string.o")
"$CLANG" "${COMMON_FLAGS[@]}" -c "$DOMAIN_SRC" -o "$OBJ_DIR/domain.o"; objs+=("$OBJ_DIR/domain.o")

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" "${objs[@]}"
"$LLVM_READOBJ" -h "$OUT_DOM" >/dev/null
[[ -f "$OUT_DOM" ]] || { echo "failed to build $OUT_DOM" >&2; exit 1; }
echo "Built $OUT_DOM"
