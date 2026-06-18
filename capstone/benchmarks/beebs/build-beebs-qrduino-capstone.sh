#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=qrduino

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
QRDUINO_DIR=$BEEBS_SRC_DIR/src/qrduino

mkdir -p "$OUT_DIR" "$OBJ_DIR"

emit_preamble() {
  printf 'typedef unsigned long size_t;\n'
  printf '#ifndef NULL\n#define NULL ((void *)0)\n#endif\n'
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
  printf 'static size_t capstone_strlen(const char *s) {\n'
  printf '  size_t n = 0;\n'
  printf '  while (s[n]) ++n;\n'
  printf '  return n;\n'
  printf '}\n'
  printf '#define memcpy capstone_memcpy\n'
  printf '#define memmove capstone_memmove\n'
  printf '#define memset capstone_memset\n'
  printf '#define strlen capstone_strlen\n'
}

strip_hosted_includes() {
  sed -E '/^#include <(string|stdlib|stddef)\.h>/d' "$1"
}

patch_qrtest() {
  {
    emit_preamble
    strip_hosted_includes "$QRDUINO_DIR/qrtest.c" |
      sed \
        -e 's/static const char \*in_encode = "http:\/\/www\.mageec\.com";/static const char in_encode[] = "http:\/\/www.mageec.com";/' \
        -e '/int verify_benchmark(int unused)/,$d'
    printf 'int verify_benchmark(int unused) {\n'
    printf '  (void)unused;\n'
    printf '  static const unsigned char expected[22] = {\n'
    printf '    254,101,63,128,130,110,160,128,186,65,46,\n'
    printf '    128,186,38,46,128,186,9,174,128,130,20\n'
    printf '  };\n'
    printf '  for (int i = 0; i < 22; i++)\n'
    printf '    if (strinbuf[i] != expected[i])\n'
    printf '      return 0;\n'
    printf '  return 1;\n'
    printf '}\n'
  } > "$OUT_DIR/qrduino_qrtest.c"
}

patch_qrencode() {
  {
    emit_preamble
    strip_hosted_includes "$QRDUINO_DIR/qrencode.c"
  } > "$OUT_DIR/qrduino_qrencode.c"
}

patch_qrframe() {
  {
    emit_preamble
    strip_hosted_includes "$QRDUINO_DIR/qrframe.c" |
      sed \
        -e 's/static char heap\[HEAP_SIZE\];/static char heap[HEAP_SIZE] __attribute__((aligned(16)));/' \
        -e '/void \*new_ptr = heap_ptr;/i\    size = (size + 15) \& ~(size_t)15;'
  } > "$OUT_DIR/qrduino_qrframe.c"
}

patch_qrtest
patch_qrencode
patch_qrframe

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
  -I"$QRDUINO_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" -c "$OUT_DIR/qrduino_qrtest.c" -o "$OBJ_DIR/qrtest.o"
"$CLANG" "${COMMON_FLAGS[@]}" -c "$OUT_DIR/qrduino_qrencode.c" -o "$OBJ_DIR/qrencode.o"
"$CLANG" "${COMMON_FLAGS[@]}" -c "$OUT_DIR/qrduino_qrframe.c" -o "$OBJ_DIR/qrframe.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_simple_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" "$OBJ_DIR/qrtest.o" "$OBJ_DIR/qrencode.o" \
  "$OBJ_DIR/qrframe.o" "$OBJ_DIR/beebs_simple_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"
echo "Built $OUT_DOM"
