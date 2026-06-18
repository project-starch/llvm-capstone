#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=ctl-string

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
BENCH_SRC=$BEEBS_SRC_DIR/src/ctl-string/string.c
PATCHED_SRC=$OUT_DIR/ctl-string_src.c

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# ctl-string uses hosted libc declarations and helpers. Keep the freestanding
# adaptation local to the generated scratch source.
{
  printf 'typedef unsigned long size_t;\n'
  printf '#ifndef NULL\n#define NULL ((void *)0)\n#endif\n'
  printf '/* Integer GROWFACTOR avoids float arithmetic in CTL_GROW_ALLOC_SIZE. */\n'
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
  printf 'static size_t capstone_strlen(const char *s) {\n'
  printf '  size_t n = 0;\n'
  printf '  while (s[n]) ++n;\n'
  printf '  return n;\n'
  printf '}\n'
  printf 'static char *capstone_strcpy(char *d, const char *s) {\n'
  printf '  char *r = d;\n'
  printf '  while ((*d++ = *s++)) {}\n'
  printf '  return r;\n'
  printf '}\n'
  printf 'static char *capstone_strncpy(char *d, const char *s, size_t n) {\n'
  printf '  size_t i = 0;\n'
  printf '  for (; i < n && s[i]; ++i) d[i] = s[i];\n'
  printf '  for (; i < n; ++i) d[i] = 0;\n'
  printf '  return d;\n'
  printf '}\n'
  printf 'static int capstone_strcmp(const char *a, const char *b) {\n'
  printf '  while (*a && *a == *b) { ++a; ++b; }\n'
  printf '  return (unsigned char)*a - (unsigned char)*b;\n'
  printf '}\n'
  printf 'static char *capstone_strchr(const char *s, int c) {\n'
  printf '  while (*s) { if (*s == (char)c) return (char *)s; ++s; }\n'
  printf '  return c == 0 ? (char *)s : (char *)0;\n'
  printf '}\n'
  printf 'static char *capstone_strrchr(const char *s, int c) {\n'
  printf '  const char *last = 0;\n'
  printf '  do { if (*s == (char)c) last = s; } while (*s++);\n'
  printf '  return (char *)last;\n'
  printf '}\n'
  printf 'static char *capstone_strstr(const char *h, const char *n) {\n'
  printf '  if (!*n) return (char *)h;\n'
  printf '  for (; *h; ++h) {\n'
  printf '    const char *a = h; const char *b = n;\n'
  printf '    while (*a && *b && *a == *b) { ++a; ++b; }\n'
  printf '    if (!*b) return (char *)h;\n'
  printf '  }\n'
  printf '  return (char *)0;\n'
  printf '}\n'
  printf 'static int capstone_tolower(int c) {\n'
  printf "  return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;\n"
  printf '}\n'
  printf '#define memcpy capstone_memcpy\n'
  printf '#define memmove capstone_memmove\n'
  printf '#define strlen capstone_strlen\n'
  printf '#define strcpy capstone_strcpy\n'
  printf '#define strncpy capstone_strncpy\n'
  printf '#define strcmp capstone_strcmp\n'
  printf '#define strchr capstone_strchr\n'
  printf '#define strrchr capstone_strrchr\n'
  printf '#define strstr capstone_strstr\n'
  printf '#define tolower capstone_tolower\n'
  sed -E '/^#include <(stdio|stdlib|string|ctype|stddef)\.h>/d' "$BENCH_SRC" |
    sed \
      -e 's/static char heap\[HEAP_SIZE\];/static char heap[HEAP_SIZE] __attribute__((aligned(16)));/' \
      -e '/void \*new_ptr = heap_ptr;/a\    size = (size + 15) \& ~(size_t)15;'
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
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_SRC" -o "$OBJ_DIR/ctl-string.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_simple_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" "$OBJ_DIR/ctl-string.o" "$OBJ_DIR/beebs_simple_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"
echo "Built $OUT_DOM"
