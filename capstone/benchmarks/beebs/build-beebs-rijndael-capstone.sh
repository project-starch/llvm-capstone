#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

BEEBS_BENCHMARK=rijndael
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
RIJ_DIR=$BEEBS_SRC_DIR/src/rijndael
AES_SRC=$RIJ_DIR/aes.c
AESXAM_SRC=$RIJ_DIR/aesxam.c
PATCHED_AES=$OUT_DIR/${BEEBS_BENCHMARK}_aes.c
PATCHED_AESXAM=$OUT_DIR/${BEEBS_BENCHMARK}_aesxam.c

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# aes.c line 538: kt = kf + nc*(Nrnd+1) - Nkey
# Bug #12: DAGCombiner folds add(ptr,neg(x)) → sub(ptr,x), backend has no sub i128.
# Fix: compute net integer offset in a local long so the GEP uses one positive-or-signed index.
sed 's/kt = kf + nc \* (cx->Nrnd + 1) - cx->Nkey;/{long _koff=(long)(nc*(cx->Nrnd+1))-(long)(cx->Nkey); kt=kf+_koff;}/' \
  "$AES_SRC" > "$PATCHED_AES"

# aesxam.c: strip hosted includes; fpos_t is self-defined but needs size_t first.
# Patches:
#   1. 'char *presetkey' → 'const char presetkey[]': global pointer would be
#      stored as an untagged raw address in .data (no ELF capability relocations);
#      an array decays to a capability at reference time via auipc+cincoffset.
#   2. 'aes ctx = {0}' → 'aes ctx': the -O0 zero-init loop for this ~1048-byte
#      struct starts the iteration pointer at null instead of &ctx (compiler bug),
#      causing cincoffsetimm to fail on the untagged null. set_key() initialises
#      all required fields before ctx is used, so the zero-init is safe to drop.
#   3. 'static char r[4]' → 'static char r[8]': fillrand() does
#      '*(unsigned long*)r = RAND(...)', an 8-byte store through a 4-byte array
#      (the code assumes sizeof(unsigned long)==4, false on rv64). This is a
#      genuine out-of-bounds write that object-granularity capability narrowing
#      (-capstone-shrink-globals) correctly traps; under broad bounds it silently
#      clobbered 4 adjacent bytes. Only r[0..3] are ever read, so widening r to
#      8 bytes makes the write in-bounds without changing the RAND stream.
{
  printf 'typedef unsigned long size_t;\n'
  sed -E '/^#include <(stdio|stdlib|ctype)\.h>/d' "$AESXAM_SRC" \
    | sed 's/^char \*presetkey=/const char presetkey[]=/' \
    | sed 's/aes     ctx = {0};/aes     ctx;/' \
    | sed -E 's/(static[[:space:]]+char[[:space:]]+)r\[4\]/\1r[8]/'
} > "$PATCHED_AESXAM"

# aes.h: 'typedef unsigned long word' with the comment "must be a 32-bit storage
# unit". On rv64 `unsigned long` is 8 bytes, so word_in(x) = *(word*)(x) reads
# 8 bytes where the algorithm intends 4. In encrypt()/decrypt(), si(...,3) does
# *(word*)(in_blk + 12), an 8-byte load at offset 12 of the 16-byte AES block --
# 4 bytes past the end (and word_out writes 8 bytes at out_blk+12 likewise).
# Under broad bounds the overlapping 8-byte-at-4-byte-stride accesses happen to
# reconstruct the right 16-byte output; object-granularity stack narrowing
# (-capstone-shrink-stack) correctly traps the over-read. Fix is the header's own
# stated intent: make `word` an actual 32-bit type (`unsigned int` is 32-bit on
# rv64), which is both in-bounds and semantically correct AES. The patched header
# is placed in OUT_DIR and shadows the fetched one via -I"$OUT_DIR".
PATCHED_AES_H=$OUT_DIR/aes.h
sed 's/typedef unsigned long   word;/typedef unsigned int    word;/' \
  "$RIJ_DIR/aes.h" > "$PATCHED_AES_H"
grep -q 'typedef unsigned int    word;' "$PATCHED_AES_H" \
  || { echo "ERROR: aes.h word typedef patch did not apply" >&2; exit 1; }

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  "$DOMAIN_OPT_LEVEL"
  -I"$OUT_DIR"
  -I"$SUPPORT_DIR"
  -I"$RIJ_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_AES"    -o "$OBJ_DIR/aes.o"
"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_AESXAM" -o "$OBJ_DIR/aesxam.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_simple_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" "$OBJ_DIR/aes.o" "$OBJ_DIR/aesxam.o" "$OBJ_DIR/beebs_simple_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"
echo "Built $OUT_DOM"
