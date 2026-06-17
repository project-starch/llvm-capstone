#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=ctl-vector

source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-$BEEBS_BENCHMARK}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$CAPSTONE_REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_${BEEBS_BENCHMARK}_capstone.dom}

SUPPORT_DIR=$BEEBS_SRC_DIR/support
CTL_SRC_DIR=$BEEBS_SRC_DIR/src/ctl-vector
PATCHED_INC=$OUT_DIR/include-ctl-vector
PATCHED_SRC=$OUT_DIR/ctl-vector_src.c

mkdir -p "$OUT_DIR" "$OBJ_DIR" "$PATCHED_INC"

# ctl.h has hosted includes unavailable in the freestanding environment.
# Also define memcpy/memmove stubs before vector.h expands macros that call them.
# CTL_GROWFACTOR is redefined as 1 (integer) to avoid the default 0.7 float
# constant, which would generate soft-float libcalls on Capstone.
{
  printf 'typedef unsigned long size_t;\n'
  printf '#ifndef NULL\n#define NULL ((void *)0)\n#endif\n'
  printf '#ifndef bool\ntypedef int bool;\n#define true 1\n#define false 0\n#endif\n'
  printf '/* Integer GROWFACTOR avoids float arithmetic in CTL_GROW_ALLOC_SIZE */\n'
  printf '#define CTL_GROWFACTOR 1\n'
  printf 'static void *capstone_memcpy(void *d, const void *s, size_t n) {\n'
  printf '  char *dd = d; const char *ss = s;\n'
  printf '  while (n--) *dd++ = *ss++;\n'
  printf '  return d;\n'
  printf '}\n'
  printf 'static void *capstone_memmove(void *d, const void *s, size_t n) {\n'
  printf '  char *dd = d; const char *ss = s;\n'
  printf '  if (dd < ss) { while (n--) *dd++ = *ss++; }\n'
  printf '  else { dd += n; ss += n; while (n--) *--dd = *--ss; }\n'
  printf '  return d;\n'
  printf '}\n'
  printf '#define memcpy  capstone_memcpy\n'
  printf '#define memmove capstone_memmove\n'
  sed -E '/^#include <(stdlib|stdbool|string|ctype|stdio)\.h>/d' \
    "$CTL_SRC_DIR/ctl.h"
} > "$PATCHED_INC/ctl.h"
cp "$CTL_SRC_DIR/vector.h" "$PATCHED_INC/vector.h"

# ctl.c: prepend size_t/NULL stubs and strip <stddef.h>
{
  printf 'typedef unsigned long size_t;\n'
  printf '#ifndef NULL\n#define NULL ((void *)0)\n#endif\n'
  sed '/^#include <stddef\.h>/d' "$CTL_SRC_DIR/ctl.c"
} > "$PATCHED_SRC"

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  "$DOMAIN_OPT_LEVEL"
  -DCTL_VECTOR
  -I"$PATCHED_INC"   # patched ctl.h + copied vector.h (before CTL_SRC_DIR)
  -I"$SUPPORT_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_SRC" -o "$OBJ_DIR/ctl-vector.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_simple_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" "$OBJ_DIR/ctl-vector.o" "$OBJ_DIR/beebs_simple_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"
echo "Built $OUT_DOM"
