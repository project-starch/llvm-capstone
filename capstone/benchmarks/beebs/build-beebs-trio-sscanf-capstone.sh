#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=trio-sscanf

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
TRIO_DIR=$BEEBS_SRC_DIR/src/trio
SCRATCH_DIR=$OUT_DIR/trio-sscanf-src
ADAPTED_DIR=$SCRIPT_DIR/adapted

mkdir -p "$OUT_DIR" "$OBJ_DIR" "$SCRATCH_DIR"

strip_hosted_includes() {
  sed -E '/^[[:space:]]*#[[:space:]]*include[[:space:]]+<(stdio|stdlib|assert|string|ctype|errno|stdint|inttypes|unistd|math|float|limits)\.h>/d' "$1"
}

for f in trio.c trio.h trio_test.c triostr.c triostr.h triodef.h triop.h; do
  strip_hosted_includes "$TRIO_DIR/$f" > "$SCRATCH_DIR/$f"
done

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  "$DOMAIN_OPT_LEVEL"
  -include "$ADAPTED_DIR/beebs_trio_capstone_preamble.h"
  -I"$SUPPORT_DIR"
  -I"$SCRATCH_DIR"
  -I"$ADAPTED_DIR"
  -DTRIO_SSCANF
  -DTRIO_EXTENSION=0
  -DTRIO_DEPRECATED=0
  -DTRIO_MICROSOFT=0
  -DTRIO_ERRORS=0
  -DTRIO_FEATURE_FLOAT=0
  -DTRIO_FEATURE_FILE=0
  -DTRIO_FEATURE_STDIO=0
  -DTRIO_FEATURE_FD=0
  -DTRIO_FEATURE_DYNAMICSTRING=0
  -DTRIO_FEATURE_CLOSURE=0
  -DTRIO_FEATURE_STRERR=0
  -DTRIO_EMBED_STRING
  -DTRIO_FUNC_LENGTH
  -DTRIO_FUNC_LENGTH_MAX
  -DTRIO_FUNC_COPY_MAX
  -DTRIO_FUNC_EQUAL
  -DTRIO_FUNC_EQUAL_CASE
  -DTRIO_FUNC_EQUAL_MAX
  -DTRIO_FUNC_TO_LONG
  -DTRIO_FUNC_TO_UNSIGNED_LONG
  -DTRIO_FUNC_TO_UPPER
  -Wno-incompatible-library-redeclaration
  -Wno-builtin-requires-header
  -Wno-implicit-function-declaration
  -Wno-macro-redefined
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" -c "$SCRATCH_DIR/trio.c" -o "$OBJ_DIR/trio.o"
"$CLANG" "${COMMON_FLAGS[@]}" -c "$SCRATCH_DIR/triostr.c" -o "$OBJ_DIR/triostr.o"
"$CLANG" "${COMMON_FLAGS[@]}" -c "$SCRATCH_DIR/trio_test.c" -o "$OBJ_DIR/trio_test.o"
"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$ADAPTED_DIR/beebs_trio_capstone_stubs.c" \
  -o "$OBJ_DIR/trio_stubs.o"
"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_simple_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/trio.o" \
  "$OBJ_DIR/triostr.o" \
  "$OBJ_DIR/trio_test.o" \
  "$OBJ_DIR/trio_stubs.o" \
  "$OBJ_DIR/beebs_simple_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"
echo "Built $OUT_DOM"
