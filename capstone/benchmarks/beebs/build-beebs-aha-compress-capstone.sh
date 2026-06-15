#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-aha-compress}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_aha-compress_capstone.dom}

AHA_SRC=$BEEBS_SRC_DIR/src/aha-compress/compress_test.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_AHA_SRC=$OUT_DIR/beebs_aha_compress_capstone.c
AHA_TAIL_SRC=$SCRIPT_DIR/adapted/beebs_aha_compress_capstone_tail.c

if [[ ! -f "$AHA_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS aha-compress source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

if [[ ! -f "$AHA_TAIL_SRC" ]]; then
  echo "missing adapted aha-compress tail source: $AHA_TAIL_SRC" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Strip stdio/stdlib includes and everything from the test[] global array
# onwards; the tail provides a static-linked replacement to avoid clashing
# with the 'test' entry-point symbol in start.S.
awk '
  /^#include <(stdio|stdlib)\.h>$/ { next }
  /^const unsigned long test\[\] = \{/ { exit }
  { print }
' "$AHA_SRC" > "$PATCHED_AHA_SRC"
cat "$AHA_TAIL_SRC" >> "$PATCHED_AHA_SRC"

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
  -c "$PATCHED_AHA_SRC" \
  -o "$OBJ_DIR/beebs_aha_compress.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_aha_compress_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_aha_compress.o" \
  "$OBJ_DIR/beebs_aha_compress_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
