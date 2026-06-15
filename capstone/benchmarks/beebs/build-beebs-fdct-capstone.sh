#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_fdct_capstone.dom}

FDCT_SRC=$BEEBS_SRC_DIR/src/fdct/libfdct.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_FDCT_SRC=$OUT_DIR/libfdct_capstone.c
FDCT_TAIL_SRC=$SCRIPT_DIR/adapted/beebs_fdct_capstone_tail.c

if [[ ! -f "$FDCT_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS fdct source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

if [[ ! -f "$FDCT_TAIL_SRC" ]]; then
  echo "missing adapted fdct tail source: $FDCT_TAIL_SRC" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Avoid hosted memcpy/memcmp in the upstream benchmark and verifier while
# keeping the fixed-point DCT kernel and expected data unchanged.
awk '
  /^void$/ { pending = $0; next }
  pending == "void" && /^initialise_benchmark / { exit }
  pending != "" { print pending; pending = "" }
  { print }
' "$FDCT_SRC" > "$PATCHED_FDCT_SRC"
cat "$FDCT_TAIL_SRC" >> "$PATCHED_FDCT_SRC"

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
  -c "$PATCHED_FDCT_SRC" \
  -o "$OBJ_DIR/beebs_fdct.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_fdct_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_fdct.o" \
  "$OBJ_DIR/beebs_fdct_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
