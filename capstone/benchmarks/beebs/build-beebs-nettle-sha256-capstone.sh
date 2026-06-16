#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

BEEBS_BENCHMARK=nettle-sha256
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
NS_DIR=$BEEBS_SRC_DIR/src/nettle-sha256
NS_SRC=$NS_DIR/nettle-sha256.c
PATCHED_NS=$OUT_DIR/${BEEBS_BENCHMARK}_src.c

if [[ ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS support tree: $BEEBS_SRC_DIR" >&2
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
  -I"$NS_DIR"
)

# Prepend freestanding type/stub header; strip hosted includes.
# Patches applied:
#   1. Replace unreachable default:abort() with break (leftover=0 for length=32)
#   2. Bypass function-pointer struct: call sha256_init/update/digest directly.
#      In Capstone domains, function pointers in .rodata have no capability tags
#      (no ELF relocations), so loading them with ldc and calling with cjalr faults.
{
  printf 'typedef unsigned char uint8_t;\n'
  printf 'typedef unsigned int uint32_t;\n'
  printf 'typedef unsigned long long uint64_t;\n'
  printf 'typedef unsigned long size_t;\n'
  printf 'typedef int bool;\n'
  printf '#define true 1\n'
  printf '#define false 0\n'
  printf '#define assert(x) ((void)(x))\n'
  printf 'void *memcpy(void *dst, const void *src, size_t n);\n'
  printf 'void *memset(void *dst, int c, size_t n);\n'
  sed -E '/^#include <(stdint|stddef|stdlib|string|stdbool|assert)\.h>/d' "$NS_SRC" \
    | sed 's/          abort();/          break;/' \
    | sed 's/nettle_sha256\.init (&ctx)/sha256_init (\&ctx)/' \
    | sed 's/nettle_sha256\.update (&ctx, sizeof (msg), msg)/sha256_update (\&ctx, sizeof (msg), msg)/' \
    | sed 's/nettle_sha256\.digest (&ctx, nettle_sha256\.digest_size, buffer)/sha256_digest (\&ctx, SHA256_DIGEST_SIZE, buffer)/'
} > "$PATCHED_NS"

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_NS" -o "$OBJ_DIR/nettle-sha256.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/adapted/beebs_nettle_sha256_stubs.c" \
  -o "$OBJ_DIR/stubs.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_simple_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/nettle-sha256.o" \
  "$OBJ_DIR/stubs.o" \
  "$OBJ_DIR/beebs_simple_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"
echo "Built $OUT_DOM"
