#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=slre

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
SLRE_DIR=$BEEBS_SRC_DIR/src/slre
SLRE_TAIL=$SCRIPT_DIR/adapted/beebs_slre_capstone_tail.c
PATCHED_SRC=$OUT_DIR/slre_capstone.c

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Generate patched slre source:
# - freestanding type and libc-stub preamble
# - upstream libslre.c with hosted includes stripped and benchmark tail removed
# - adapted benchmark tail (avoids char *regexes[] — untagged pointers)
{
  printf 'typedef unsigned long size_t;\n'
  printf '#ifndef NULL\n#define NULL ((void *)0)\n#endif\n'
  printf 'static size_t strlen(const char *s) {\n'
  printf '  size_t n = 0; while (s[n]) ++n; return n;\n'
  printf '}\n'
  printf 'static int memcmp(const void *s1, const void *s2, size_t n) {\n'
  printf '  const unsigned char *a = s1, *b = s2;\n'
  printf '  for (; n; --n, ++a, ++b) {\n'
  printf '    if (*a < *b) return -1;\n'
  printf '    if (*a > *b) return  1;\n'
  printf '  }\n'
  printf '  return 0;\n'
  printf '}\n'
  printf 'static int isdigit(int c) { return c >= 48 && c <= 57; }\n'
  printf 'static int isxdigit(int c) {\n'
  printf '  return (c >= 48 && c <= 57) || (c >= 65 && c <= 70) || (c >= 97 && c <= 102);\n'
  printf '}\n'
  printf 'static int isspace(int c) { return c == 32 || (c >= 9 && c <= 13); }\n'
  printf 'static int tolower(int c) { return (c >= 65 && c <= 90) ? c + 32 : c; }\n'
  printf 'static const char *strchr(const char *s, int c) {\n'
  printf '  while (*s) { if (*s == (char)c) return s; ++s; }\n'
  printf '  return (char)c == 0 ? s : (const char *)0;\n'
  printf '}\n'
  sed -E '/^#include <(stdio|ctype|string)\.h>/d' "$SLRE_DIR/libslre.c" |
    sed '/^char text\[\]/,$d'
  cat "$SLRE_TAIL"
} > "$PATCHED_SRC"

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
  -I"$SLRE_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_SRC" -o "$OBJ_DIR/slre.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_simple_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" "$OBJ_DIR/slre.o" "$OBJ_DIR/beebs_simple_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"
echo "Built $OUT_DOM"
