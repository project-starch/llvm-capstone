#!/bin/bash
# Build the three supervised-CALL directed tests as bare M-mode board images (load at 0x80000000).
# Self-contained: env/ holds the riscv-tests headers the run was built with (capstone-ariane's riscv-tests
# checkout, env 2f75dc2 plus local modifications to p/riscv_test.h). Only the compiler is external.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd)
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
mkdir -p "$B/out"
build() {   # name test-file id-char [extra -D]
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$B/env/macros" -I"$B/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch "$@" -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/out/$n.bin"
  printf '%-14s %8d bytes  sha256 %s\n' "$n" "$(stat -c %s "$B/out/$n.bin")" "$(sha256sum "$B/out/$n.bin" | cut -c1-16)"
}
build sup-quantum sup-quantum.S 81
build sup-escape  sup-escape.S  69
build call-retpc  call-retpc.S  82
