#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAPSTONE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$CAPSTONE_DIR/.." && pwd)
LLVM_BIN=${LLVM_BIN:-$REPO_ROOT/llvm/build/bin}
BUILDROOT_HOST_BIN=${BUILDROOT_HOST_BIN:-$CAPSTONE_DIR/caplifive-buildroot/build/host/bin}
SRC=${1:-$SCRIPT_DIR/main.c}
DOM_OUT=${DOM_OUT:-$SCRIPT_DIR/my_domain.dom}
OBJ_DIR=${OBJ_DIR:-$SCRIPT_DIR/.build}

CLANG=${CLANG:-$LLVM_BIN/clang}
HOST_LD=${HOST_LD:-$LLVM_BIN/ld.lld}

mkdir -p "$OBJ_DIR"

START_O="$OBJ_DIR/start.o"
MAIN_O="$OBJ_DIR/main.o"

"$CLANG" -target capstone64-unknown-elf -ffreestanding -c "$SCRIPT_DIR/start.S" -o "$START_O"
"$CLANG" -target capstone64-unknown-elf -ffreestanding -O0 -c "$SRC" -o "$MAIN_O"

if [[ "$(basename -- "$HOST_LD")" != "ld.lld" ]]; then
  # Compatibility shim for external RISC-V linkers that do not understand
  # EM_CAPSTONE yet.
  python3 - "$START_O" "$MAIN_O" <<'PY'
from pathlib import Path
import sys

EM_RISCV = (243).to_bytes(2, "little")

for arg in sys.argv[1:]:
	path = Path(arg)
	data = bytearray(path.read_bytes())
	if data[:4] != b"\x7fELF":
		raise SystemExit(f"{path}: not an ELF file")
	data[18:20] = EM_RISCV
	path.write_bytes(data)
PY
fi

"$HOST_LD" -T "$SCRIPT_DIR/link.ld" -o "$DOM_OUT" "$START_O" "$MAIN_O"

echo "Built $DOM_OUT"
