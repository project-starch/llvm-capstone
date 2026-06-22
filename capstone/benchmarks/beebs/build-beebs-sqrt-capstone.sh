#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-sqrt}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_sqrt_capstone.dom}

SQRT_SRC=$BEEBS_SRC_DIR/src/sqrt/libsqrt.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
COMPILER_RT=$REPO_ROOT/compiler-rt/lib/builtins

if [[ ! -f "$SQRT_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS sqrt source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

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

"$CLANG" "${COMMON_FLAGS[@]}" -c "$SQRT_SRC" -o "$OBJ_DIR/libsqrt.o"
objs+=("$OBJ_DIR/libsqrt.o")

source "$SCRIPT_DIR/build-beebs-softfloat-common.sh"
objs+=("${softfloat_objs[@]}")

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" -o "$OBJ_DIR/beebs_sqrt_domain.o"
objs+=("$OBJ_DIR/beebs_sqrt_domain.o")

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" "${objs[@]}"

"$LLVM_READOBJ" -h "$OUT_DOM" >/dev/null

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
