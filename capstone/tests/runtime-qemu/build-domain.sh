#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
LLVM_BIN=${LLVM_BIN:-$REPO_ROOT/llvm/cmake-build-debug/bin}
CLANG=${CLANG:-$LLVM_BIN/clang}
LD_LLD=${LD_LLD:-$LLVM_BIN/ld.lld}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <domain_main.c> <output.dom>" >&2
  exit 1
fi

SRC=$1
OUT=$2
OBJ_DIR=${OBJ_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/capstone-domain-build.XXXXXX")}
START_O="$OBJ_DIR/start.o"
MAIN_O="$OBJ_DIR/main.o"

mkdir -p "$(dirname -- "$OUT")"

"$CLANG" -target capstone64-unknown-elf -ffreestanding -c "$START_SRC" -o "$START_O"
"$CLANG" -target capstone64-unknown-elf -ffreestanding -O0 -c "$SRC" -o "$MAIN_O"
"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT" "$START_O" "$MAIN_O"

echo "Built $OUT"

