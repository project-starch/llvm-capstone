#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-nbody}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_nbody_capstone.dom}

NBODY_SRC=$BEEBS_SRC_DIR/src/nbody/nbody.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
COMPILER_RT=$REPO_ROOT/compiler-rt/lib/builtins
LIBM_SRC=$SCRIPT_DIR/adapted/beebs_softfloat_libm.c
PATCHED_NBODY_SRC=$OUT_DIR/libnbody_capstone.c

if [[ ! -f "$NBODY_SRC" || ! -f "$SUPPORT_DIR/support.h" || ! -f "$LIBM_SRC" ]]; then
  echo "missing required nbody sources" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Strip the hosted math/stdlib headers; provide a freestanding prototype. sqrt
# is supplied by the shared soft-float libm. (nbody references only soft-float +
# sqrt; nothing from stdlib.) The expected[] table is already `static`, so no
# Bug #9 stack memcpy is emitted.
{
  printf 'double sqrt(double);\n'
  sed -e '/#include <math.h>/d' -e '/#include <stdlib.h>/d' "$NBODY_SRC"
} > "$PATCHED_NBODY_SRC"

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

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_NBODY_SRC" -o "$OBJ_DIR/libnbody.o"
objs+=("$OBJ_DIR/libnbody.o")

"$CLANG" "${COMMON_FLAGS[@]}" -c "$LIBM_SRC" -o "$OBJ_DIR/softfloat_libm.o"
objs+=("$OBJ_DIR/softfloat_libm.o")

source "$SCRIPT_DIR/build-beebs-softfloat-common.sh"
objs+=("${softfloat_objs[@]}")

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" -o "$OBJ_DIR/beebs_nbody_domain.o"
objs+=("$OBJ_DIR/beebs_nbody_domain.o")

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" "${objs[@]}"

"$LLVM_READOBJ" -h "$OUT_DOM" >/dev/null

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
