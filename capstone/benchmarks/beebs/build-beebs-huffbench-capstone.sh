#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

BEEBS_BENCHMARK=huffbench
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
HB_SRC=$BEEBS_SRC_DIR/src/huffbench/libhuffbench.c
PATCHED_HB=$OUT_DIR/${BEEBS_BENCHMARK}_src.c
HB_PREFIX_SRC=$SCRIPT_DIR/adapted/beebs_huffbench_capstone_prefix.c
HB_RANDOM4_SRC=$SCRIPT_DIR/adapted/beebs_huffbench_random4_capstone.c

if [[ ! -f "$SUPPORT_DIR/support.h" || ! -f "$HB_SRC" ]]; then
  echo "missing BEEBS support tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

if [[ ! -f "$HB_PREFIX_SRC" || ! -f "$HB_RANDOM4_SRC" ]]; then
  echo "missing adapted huffbench source snippets" >&2
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
)

# Prepend freestanding type stubs; strip hosted includes; replace random4()
# with an adapted implementation that avoids long integer division (Capstone
# backend bug). verify_benchmark() returns -1, so RNG identity is irrelevant.
cp "$HB_PREFIX_SRC" "$PATCHED_HB"
awk -v random4_src="$HB_RANDOM4_SRC" '
  /^#include <(string|stdio|stdlib|stddef|stdbool|math)\.h>$/ { next }
  /^static size_t random4\(\)$/ {
    while ((getline line < random4_src) > 0)
      print line
    close(random4_src)
    skip_random4 = 1
    brace_depth = 0
    next
  }
  skip_random4 {
    brace_depth += gsub(/\{/, "{")
    brace_depth -= gsub(/\}/, "}")
    if (brace_depth == 0 && $0 ~ /^}$/)
      skip_random4 = 0
    next
  }
  { print }
' "$HB_SRC" >> "$PATCHED_HB"

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED_HB" -o "$OBJ_DIR/huffbench.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/adapted/beebs_huffbench_stubs.c" \
  -o "$OBJ_DIR/stubs.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_simple_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/huffbench.o" \
  "$OBJ_DIR/stubs.o" \
  "$OBJ_DIR/beebs_simple_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"
echo "Built $OUT_DOM"
