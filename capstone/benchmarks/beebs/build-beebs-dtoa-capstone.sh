#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-dtoa}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_dtoa_capstone.dom}

DTOA_SRC=$BEEBS_SRC_DIR/src/dtoa/libdtoa.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
COMPILER_RT=$REPO_ROOT/compiler-rt/lib/builtins
STRING_SRC=$SCRIPT_DIR/adapted/beebs_freestanding_string.c
LIBM_SRC=$SCRIPT_DIR/adapted/beebs_softfloat_libm.c
PRELUDE=$SCRIPT_DIR/adapted/beebs_dtoa_freestanding_prelude.h
PATCHED_SRC=$OUT_DIR/libdtoa_capstone.c

for f in "$DTOA_SRC" "$SUPPORT_DIR/support.h" "$STRING_SRC" "$LIBM_SRC" "$PRELUDE"; do
  if [[ ! -f "$f" ]]; then
    echo "missing required source: $f" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Build the patched source:
#  - prepend the freestanding prelude (size_t/NULL + string/libm protos + errno);
#  - strip the hosted quoted includes the prelude replaces (keep "float.h" --
#    clang provides a freestanding one; DEBUG/USE_LOCALE/Honor_FLT_ROUNDS includes
#    are behind off #ifdefs);
#  - blocker #2 fix (arena 16-byte alignment): the upstream `malloc_beebs` bump
#    allocator hands out `Bigint`s whose first field is a 16-byte capability, so
#    each allocation must be 16-aligned or the capability loses its tag.  With
#    -DOmit_Private_Memory every Bigint comes from malloc_beebs; we 16-align the
#    heap and round each request up to a 16-byte multiple (integer rounding only --
#    no pointer forging), keeping heap_ptr 16-aligned across allocations.  Also
#    enlarge HEAP_SIZE since all allocations now come from this one pool.
{
  cat "$PRELUDE"
  sed -e '/#include "stdlib.h"/d' \
      -e '/#include "string.h"/d' \
      -e '/#include "errno.h"/d' \
      -e '/#include "math.h"/d' \
      -e 's|#define HEAP_SIZE 8192|#define HEAP_SIZE 65536|' \
      -e 's|static char heap\[HEAP_SIZE\];|static char heap[HEAP_SIZE] __attribute__((aligned(16)));|' \
      -e 's|    void \*new_ptr = heap_ptr;|    size = (size + 15u) \& ~(size_t)15u;\n    void *new_ptr = heap_ptr;|' \
      "$DTOA_SRC"
} > "$PATCHED_SRC"

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
  -DLong=int
  -DNO_HEX_FP
  -DOmit_Private_Memory
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$START_SRC" -o "$OBJ_DIR/start.o"

objs=("$OBJ_DIR/start.o")

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_SRC" -o "$OBJ_DIR/libdtoa.o"
objs+=("$OBJ_DIR/libdtoa.o")

"$CLANG" "${COMMON_FLAGS[@]}" -c "$STRING_SRC" -o "$OBJ_DIR/dtoa_string.o"
objs+=("$OBJ_DIR/dtoa_string.o")

"$CLANG" "${COMMON_FLAGS[@]}" -c "$LIBM_SRC" -o "$OBJ_DIR/dtoa_libm.o"
objs+=("$OBJ_DIR/dtoa_libm.o")

source "$SCRIPT_DIR/build-beebs-softfloat-common.sh"
objs+=("${softfloat_objs[@]}")

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" -o "$OBJ_DIR/beebs_dtoa_domain.o"
objs+=("$OBJ_DIR/beebs_dtoa_domain.o")

# --gc-sections drops the unreferenced dtoa() output path and any string/libm
# routines strtod does not pull in.
"$LD_LLD" -T "$LINKER_SCRIPT" --gc-sections -o "$OUT_DOM" "${objs[@]}"

"$LLVM_READOBJ" -h "$OUT_DOM" >/dev/null

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
