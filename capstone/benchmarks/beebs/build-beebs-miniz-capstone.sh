#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=miniz

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
MINIZ_DIR=$BEEBS_SRC_DIR/src/miniz

mkdir -p "$OUT_DIR" "$OBJ_DIR"

emit_preamble() {
  printf 'typedef unsigned long size_t;\n'
  printf '#ifndef NULL\n#define NULL ((void *)0)\n#endif\n'
  printf '#define MINIZ_NO_MALLOC\n'
  printf '#define assert(x) ((void)(x))\n'
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
  printf 'static void *capstone_memset(void *d, int c, size_t n) {\n'
  printf '  unsigned char *dd = d;\n'
  printf '  while (n--) *dd++ = (unsigned char)c;\n'
  printf '  return d;\n'
  printf '}\n'
  printf 'static int capstone_memcmp(const void *s1, const void *s2, size_t n) {\n'
  printf '  const unsigned char *a = s1; const unsigned char *b = s2;\n'
  printf '  for (; n; --n, ++a, ++b) {\n'
  printf '    if (*a < *b) return -1;\n'
  printf '    if (*a > *b) return 1;\n'
  printf '  }\n'
  printf '  return 0;\n'
  printf '}\n'
  printf 'static size_t capstone_strlen(const char *s) {\n'
  printf '  size_t n = 0;\n'
  printf '  while (s[n]) ++n;\n'
  printf '  return n;\n'
  printf '}\n'
  printf '#define memcpy capstone_memcpy\n'
  printf '#define memmove capstone_memmove\n'
  printf '#define memset capstone_memset\n'
  printf '#define memcmp capstone_memcmp\n'
  printf '#define strlen capstone_strlen\n'
}

strip_hosted_includes() {
  sed -E '/^#include <(string|stdlib|stddef|assert)\.h>/d' "$1"
}

# Generate patched miniz.c: strip hosted includes, align heap, pad malloc size.
patch_miniz_c() {
  {
    emit_preamble
    strip_hosted_includes "$MINIZ_DIR/miniz.c" |
      sed \
        -e 's/#define HEAP_SIZE 8192/#define HEAP_SIZE (256 * 1024)/' \
        -e 's/static char heap\[HEAP_SIZE\];/static char heap[HEAP_SIZE] __attribute__((aligned(16)));/' \
        -e '/void \*new_ptr = heap_ptr;/a\    size = (size + 15) \& ~(size_t)15;'
  } > "$OUT_DIR/miniz_capstone.c"
}

# Generate patched miniz_b.c: strip hosted includes, provide local libc stubs,
# and keep the benchmark text as an array-backed capability.
patch_miniz_b_c() {
  {
    emit_preamble
    strip_hosted_includes "$MINIZ_DIR/miniz_b.c" |
      sed -e 's/^const char \*text=/const char text[]=/'
  } > "$OUT_DIR/miniz_b_capstone.c"
}

patch_miniz_c
patch_miniz_b_c

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
  -I"$MINIZ_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" -c "$OUT_DIR/miniz_capstone.c" -o "$OBJ_DIR/miniz.o"
"$CLANG" "${COMMON_FLAGS[@]}" -c "$OUT_DIR/miniz_b_capstone.c" -o "$OBJ_DIR/miniz_b.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_simple_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" "$OBJ_DIR/miniz.o" "$OBJ_DIR/miniz_b.o" \
  "$OBJ_DIR/beebs_simple_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"
echo "Built $OUT_DOM"
