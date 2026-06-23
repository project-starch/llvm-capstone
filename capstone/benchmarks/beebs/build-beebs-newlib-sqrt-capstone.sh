#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-newlib-sqrt}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_newlib-sqrt_capstone.dom}

NEWLIB_SRC=$BEEBS_SRC_DIR/src/newlib-sqrt/ef_sqrt.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
COMPILER_RT=$REPO_ROOT/compiler-rt/lib/builtins
PATCHED_SRC=$OUT_DIR/ef_sqrt_capstone.c
TAIL_SRC=$SCRIPT_DIR/adapted/beebs_newlib_sqrt_capstone_tail.c

if [[ ! -f "$NEWLIB_SRC" || ! -f "$SUPPORT_DIR/support.h" || ! -f "$TAIL_SRC" ]]; then
  echo "missing required newlib-sqrt sources" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Rename the upstream verifier; the adapted tail re-emits it with `exp[]` as a
# `static const` (Bug #9: a local const array would memcpy from .rodata into a
# stack slot with an untagged destination capability). The exact `==` check is
# unchanged. __ieee754_sqrtf is correctly-rounded, so soft-float output matches.
{
  printf '#define verify_benchmark sqrt_orig_verify\n'
  cat "$NEWLIB_SRC"
  cat "$TAIL_SRC"
} > "$PATCHED_SRC"

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  -ffp-contract=off
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$START_SRC" -o "$OBJ_DIR/start.o"

objs=("$OBJ_DIR/start.o")

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_SRC" -o "$OBJ_DIR/ef_sqrt.o"
objs+=("$OBJ_DIR/ef_sqrt.o")

source "$SCRIPT_DIR/build-beebs-softfloat-common.sh"
objs+=("${softfloat_objs[@]}")

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" -o "$OBJ_DIR/beebs_newlib_sqrt_domain.o"
objs+=("$OBJ_DIR/beebs_newlib_sqrt_domain.o")

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" "${objs[@]}"

"$LLVM_READOBJ" -h "$OUT_DOM" >/dev/null

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
