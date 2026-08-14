#!/usr/bin/env bash
# Build the resumable-yield probe: a pure-capability .dom plus its Linux host.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

REPO_ROOT=$CAPSTONE_REPO_ROOT
CLANG=${CAPSTONE_CLANG:?}
LD_LLD=${CAPSTONE_LD_LLD:?}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-capstone-yield-probe}
OBJ_DIR="$OUT_DIR/obj"
OUT_DOM=${OUT_DOM:-$OUT_DIR/yield_probe.dom}
OUT_HOST=${OUT_HOST:-$OUT_DIR/yield_probe.user}

START_SRC="$SCRIPT_DIR/../runtime/start-musl.S"
LINKER_SCRIPT="$REPO_ROOT/capstone/my_first_domain/link.ld"
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_C="$REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"

mkdir -p "$OBJ_DIR"

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$START_SRC" -o "$OBJ_DIR/start-musl.o"

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -fno-builtin -fno-jump-tables -ffunction-sections -fdata-sections \
  -O1 -c "$SCRIPT_DIR/yield_probe_domain.c" -o "$OBJ_DIR/yield_probe_domain.o"

"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start-musl.o" "$OBJ_DIR/yield_probe_domain.o"

"$GUEST_CC" -O2 -o "$OUT_HOST" "$SCRIPT_DIR/yield_probe_host.c" "$LIBCAPSTONE_C"

printf 'built %s\nbuilt %s\n' "$OUT_DOM" "$OUT_HOST"
