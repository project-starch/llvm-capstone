#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-crc32}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_crc32_capstone.dom}

CRC32_SRC=$BEEBS_SRC_DIR/src/crc32/crc_32.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_CRC32_SRC=$OUT_DIR/crc_32_capstone.c
CRC32_TAIL_SRC=$SCRIPT_DIR/adapted/beebs_crc32_capstone_tail.c

if [[ ! -f "$CRC32_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS crc32 source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

if [[ ! -f "$CRC32_TAIL_SRC" ]]; then
  echo "missing adapted crc32 tail source: $CRC32_TAIL_SRC" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Capstone-specific patches:
# 1. Strip <stdlib.h> — not available freestanding (unused by benchmark code).
# 2. Change `typedef unsigned long DWORD` to `typedef unsigned int DWORD`.
#    On Capstone `unsigned long` is 64 bits.  With 64-bit DWORD the table
#    crc_32_tab has 64-bit elements, so crc_32_tab[i] accesses 8-byte-stride
#    entries while the code multiplies the index by 4 (slli 2) — wrong stride.
#    With 32-bit unsigned int DWORD, element stride is 4 bytes — correct.
# 3. Strip verify_benchmark and append the tail, which compares against the
#    single-call expected value 1703161001 (our domain calls benchmark() once,
#    while the upstream expected value 1207487004 requires 32 iterations).
awk '
  /^#include <stdlib\.h>$/ { next }
  /^int verify_benchmark/ { exit }
  { print }
' "$CRC32_SRC" > "$PATCHED_CRC32_SRC"
perl -pi \
  -e 's/typedef unsigned long\s+DWORD;/typedef unsigned int  DWORD;/' \
  "$PATCHED_CRC32_SRC"
cat "$CRC32_TAIL_SRC" >> "$PATCHED_CRC32_SRC"

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$PATCHED_CRC32_SRC" \
  -o "$OBJ_DIR/beebs_crc32.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_crc32_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_crc32.o" \
  "$OBJ_DIR/beebs_crc32_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
