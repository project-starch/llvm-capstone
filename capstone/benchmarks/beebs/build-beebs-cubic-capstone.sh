#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-cubic}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_cubic_capstone.dom}

CUBIC_SRC=$BEEBS_SRC_DIR/src/cubic/libcubic.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_CUBIC_SRC=$OUT_DIR/libcubic_capstone.c
LIBM_SRC=$SCRIPT_DIR/adapted/beebs_cubic_libm.c
TAIL_SRC=$SCRIPT_DIR/adapted/beebs_cubic_capstone_tail.c
COMPILER_RT=$REPO_ROOT/compiler-rt/lib/builtins

for f in "$CUBIC_SRC" "$SUPPORT_DIR/support.h" "$LIBM_SRC" "$TAIL_SRC"; do
  if [[ ! -f "$f" ]]; then
    echo "missing required source: $f" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Double pivot: SolveCubic's long double (-> fp128, needs quad soft-float that
# hits the i128 non-vector-shift backend limit) is reduced to double.  Strip the
# hosted math includes and provide freestanding prototypes; define PI as a
# literal so the (4*atan(1)) form does not pull in atan.
{
  printf '#define PI 3.14159265358979323846\n'
  printf 'double acos(double); double cos(double); double sin(double);\n'
  printf 'double sqrt(double); double pow(double, double); double fabs(double);\n'
  printf 'double exp(double); double log(double);\n'
  sed -e 's/long double/double/g' \
      -e '/#include <math.h>/d' \
      -e '/#include "snipmath.h"/d' \
      "$CUBIC_SRC"
} > "$PATCHED_CUBIC_SRC"

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
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

objs=("$OBJ_DIR/start.o")

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_CUBIC_SRC" -o "$OBJ_DIR/libcubic.o"
objs+=("$OBJ_DIR/libcubic.o")

"$CLANG" "${COMMON_FLAGS[@]}" -c "$LIBM_SRC" -o "$OBJ_DIR/cubic_libm.o"
objs+=("$OBJ_DIR/cubic_libm.o")

"$CLANG" "${COMMON_FLAGS[@]}" -c "$TAIL_SRC" -o "$OBJ_DIR/cubic_tail.o"
objs+=("$OBJ_DIR/cubic_tail.o")

source "$SCRIPT_DIR/build-beebs-softfloat-common.sh"
objs+=("${softfloat_objs[@]}")

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_cubic_domain.o"
objs+=("$OBJ_DIR/beebs_cubic_domain.o")

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" "${objs[@]}"

"$LLVM_READOBJ" -h "$OUT_DOM" >/dev/null

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
