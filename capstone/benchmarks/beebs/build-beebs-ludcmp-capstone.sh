#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-ludcmp}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_ludcmp_capstone.dom}

LUDCMP_SRC=$BEEBS_SRC_DIR/src/ludcmp/libludcmp.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
COMPILER_RT=$REPO_ROOT/compiler-rt/lib/builtins
PATCHED_LUDCMP_SRC=$OUT_DIR/libludcmp_capstone.c

if [[ ! -f "$LUDCMP_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing required ludcmp sources" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Bug #9 workaround: verify_benchmark's local const-initialized expected arrays
# (exp_a/exp_b/exp_x) are otherwise copied from .rodata into stack arrays via a
# memcpy whose stack destination capability comes back untagged. Marking them
# `static const` places them directly in .rodata (no stack copy, no memcpy), as
# done for mergesort / nettle-*. The matrix algorithm itself is unaffected.
sed -E 's/^  float (exp_[abx])/  static const float \1/' \
  "$LUDCMP_SRC" > "$PATCHED_LUDCMP_SRC"

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$START_SRC" -o "$OBJ_DIR/start.o"

objs=("$OBJ_DIR/start.o")

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_LUDCMP_SRC" -o "$OBJ_DIR/libludcmp.o"
objs+=("$OBJ_DIR/libludcmp.o")

source "$SCRIPT_DIR/build-beebs-softfloat-common.sh"
objs+=("${softfloat_objs[@]}")

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" -o "$OBJ_DIR/beebs_ludcmp_domain.o"
objs+=("$OBJ_DIR/beebs_ludcmp_domain.o")

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" "${objs[@]}"

"$LLVM_READOBJ" -h "$OUT_DOM" >/dev/null

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
