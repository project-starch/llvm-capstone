#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-matmult-float}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_matmult-float_capstone.dom}

MATMULT_SRC=$BEEBS_SRC_DIR/src/matmult-float/matmult.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
COMPILER_RT=$REPO_ROOT/compiler-rt/lib/builtins
PATCHED_SRC=$OUT_DIR/matmult_float_capstone.c
TAIL_SRC=$SCRIPT_DIR/adapted/beebs_matmult_float_capstone_tail.c

for f in "$MATMULT_SRC" "$SUPPORT_DIR/support.h" "$TAIL_SRC"; do
  if [[ ! -f "$f" ]]; then
    echo "missing required source: $f" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Strip <stdio.h>/<stdlib.h> (not available freestanding) and everything from
# verify_benchmark onwards; the tail provides an FNV-1a checksum of the global
# ResultArray (avoids the local float exp[][] -> Bug #3 i128 stride / Bug #9).
awk '
  /^#include <(stdio|stdlib)\.h>$/ { next }
  /^int verify_benchmark/ { exit }
  { print }
' "$MATMULT_SRC" > "$PATCHED_SRC"
cat "$TAIL_SRC" >> "$PATCHED_SRC"

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  -ffp-contract=off
  -ffunction-sections
  -fdata-sections
  -DMATMULT_FLOAT
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$START_SRC" -o "$OBJ_DIR/start.o"

objs=("$OBJ_DIR/start.o")

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_SRC" -o "$OBJ_DIR/beebs_matmult_float.o"
objs+=("$OBJ_DIR/beebs_matmult_float.o")

source "$SCRIPT_DIR/build-beebs-softfloat-common.sh"
objs+=("${softfloat_objs[@]}")

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" -o "$OBJ_DIR/beebs_matmult_float_domain.o"
objs+=("$OBJ_DIR/beebs_matmult_float_domain.o")

# --gc-sections drops the upstream `values_match` (dead once verify_benchmark is
# replaced; it would otherwise pull in undefined frexpf/fabsf libcalls).
"$LD_LLD" -T "$LINKER_SCRIPT" --gc-sections -o "$OUT_DOM" "${objs[@]}"

"$LLVM_READOBJ" -h "$OUT_DOM" >/dev/null

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
