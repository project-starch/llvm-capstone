#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

if [[ -z "${BEEBS_BENCHMARK:-}" || ${#BEEBS_SOURCE_FILES_REL[@]} -eq 0 ]]; then
  echo "BEEBS_BENCHMARK and BEEBS_SOURCE_FILES_REL must be set before sourcing this file" >&2
  exit 1
fi

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-$BEEBS_BENCHMARK}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_${BEEBS_BENCHMARK}_capstone.dom}

SUPPORT_DIR=$BEEBS_SRC_DIR/support
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
)

for include_rel in "${BEEBS_EXTRA_INCLUDE_RELS[@]:-}"; do
  COMMON_FLAGS+=("-I$BEEBS_SRC_DIR/$include_rel")
done

for def in "${BEEBS_EXTRA_DEFINES[@]:-}"; do
  [[ -n "$def" ]] && COMMON_FLAGS+=("-D$def")
done

sanitize_source() {
  local src=$1
  local dst=$2

  # Build the sed expression list
  local sed_exprs=()

  if [[ "${BEEBS_STRIP_HOSTED_INCLUDES:-0}" == 1 ]]; then
    local strip_pat='stdio|stdlib'
    for hdr in "${BEEBS_STRIP_EXTRA_HEADERS[@]:-}"; do
      [[ -n "$hdr" ]] && strip_pat="${strip_pat}|${hdr}"
    done
    sed_exprs+=(-e "/^#include <(${strip_pat})\.h>/d")
  fi

  for expr in "${BEEBS_EXTRA_SED_EXPRS[@]:-}"; do
    [[ -n "$expr" ]] && sed_exprs+=(-e "$expr")
  done

  {
    for line in "${BEEBS_PREAMBLE_LINES[@]:-}"; do
      [[ -n "$line" ]] && printf '%s\n' "$line"
    done
    if [[ "${BEEBS_DEFINE_NULL:-0}" == 1 ]]; then
      printf '#ifndef NULL\n#define NULL ((void *)0)\n#endif\n'
    fi
    if [[ ${#sed_exprs[@]} -gt 0 ]]; then
      sed -E "${sed_exprs[@]}" "$src"
    else
      cat "$src"
    fi
  } > "$dst"
}

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

objs=("$OBJ_DIR/start.o")
idx=0
for source_rel in "${BEEBS_SOURCE_FILES_REL[@]}"; do
  src=$BEEBS_SRC_DIR/$source_rel
  if [[ ! -f "$src" ]]; then
    echo "missing BEEBS source: $src" >&2
    exit 1
  fi

  patched_src=$OUT_DIR/${BEEBS_BENCHMARK}_src_${idx}.c
  sanitize_source "$src" "$patched_src"

  "$CLANG" "${COMMON_FLAGS[@]}" \
    -c "$patched_src" \
    -o "$OBJ_DIR/src-$idx.o"
  objs+=("$OBJ_DIR/src-$idx.o")
  idx=$((idx + 1))
done

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_simple_domain.o"
objs+=("$OBJ_DIR/beebs_simple_domain.o")

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" "${objs[@]}"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
