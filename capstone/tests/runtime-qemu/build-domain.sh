#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

REPO_ROOT=${CAPSTONE_REPO_ROOT}
LLVM_BIN=${LLVM_BIN:-$CAPSTONE_LLVM_BIN}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
GCT_TAIL_SRC=${GCT_TAIL_SRC:-$SCRIPT_DIR/gct-section-end.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O2}

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <domain_main.c> <output.dom>" >&2
  exit 1
fi

SRC=$1
OUT=$2
OBJ_DIR=${OBJ_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/capstone-domain-build.XXXXXX")}
START_O="$OBJ_DIR/start.o"
GCT_TAIL_O="$OBJ_DIR/gct-tail.o"
MAIN_O="$OBJ_DIR/main.o"

mkdir -p "$(dirname -- "$OUT")"

"$CLANG" -target capstone64-unknown-elf -ffreestanding -c "$START_SRC" -o "$START_O"
"$CLANG" -target capstone64-unknown-elf -ffreestanding -c "$GCT_TAIL_SRC" -o "$GCT_TAIL_O"
"$CLANG" -target capstone64-unknown-elf -ffreestanding "$DOMAIN_OPT_LEVEL" -c "$SRC" -o "$MAIN_O"
"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT" "$START_O" "$MAIN_O" "$GCT_TAIL_O"

echo "Built $OUT"

