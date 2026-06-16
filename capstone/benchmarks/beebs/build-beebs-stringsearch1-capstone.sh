#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-stringsearch1}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_stringsearch1_capstone.dom}

SS_MAIN_SRC=$BEEBS_SRC_DIR/src/stringsearch1/stringsearch1.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_MAIN_SRC=$OUT_DIR/stringsearch1_capstone.c
FWD_SRC=$SCRIPT_DIR/adapted/beebs_stringsearch1_fwd_capstone.c
REV_SRC=$SCRIPT_DIR/adapted/beebs_stringsearch1_rev_capstone.c

if [[ ! -f "$SS_MAIN_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS stringsearch1 source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

for f in "$FWD_SRC" "$REV_SRC"; do
  if [[ ! -f "$f" ]]; then
    echo "missing adapted source: $f" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Capstone-specific patches to stringsearch1.c:
#
# 1. Strip <string.h> — not available freestanding.  strlen is defined in
#    beebs_stringsearch1_fwd_capstone.c and declared extern here.
#
# 2. Add -fno-jump-tables flag: the switch in benchmark() would otherwise
#    generate a jump table using lui+addi absolute addresses (plain integers,
#    not capabilities), which faults with "Cap mem access requires capability".
#    (stringsearch1.c has no switch, but the flag is kept for consistency.)
awk '
  /^#include <string\.h>$/ {
    print "extern unsigned long strlen(const char *s);"
    next
  }
  { print }
' "$SS_MAIN_SRC" > "$PATCHED_MAIN_SRC"

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

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$PATCHED_MAIN_SRC" \
  -o "$OBJ_DIR/beebs_stringsearch1_main.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$FWD_SRC" \
  -o "$OBJ_DIR/beebs_stringsearch1_fwd.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$REV_SRC" \
  -o "$OBJ_DIR/beebs_stringsearch1_rev.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_stringsearch1_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_stringsearch1_main.o" \
  "$OBJ_DIR/beebs_stringsearch1_fwd.o" \
  "$OBJ_DIR/beebs_stringsearch1_rev.o" \
  "$OBJ_DIR/beebs_stringsearch1_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
