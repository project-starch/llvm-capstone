#!/bin/bash
# Build cap-atomics as a bare M-mode board image (load at 0x80000000), with the supervised-CALL ladder's committed
# board harness (../sup-bare-2026-10-03-ladder: inc/ = UART prologue + CAPPRINT recorder, env/ = riscv-tests headers).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd)
H=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
mkdir -p "$B/out"
"$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
  -fuse-ld=lld -I"$H/inc" -I"$B/tests" -I"$H/env/macros" -I"$H/env/p" \
  -DBOARD_TEST_FILE='"cap-atomics.S"' -DBOARD_TEST_CH=65 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8 \
  -T "$H/inc/board.ld" "$H/inc/board_wrap.S" -o "$B/out/cap-atomics.elf"
"$BIN/llvm-objcopy" -O binary "$B/out/cap-atomics.elf" "$B/out/cap-atomics.bin"
printf 'cap-atomics %8d bytes  sha256 %s  board_rec %s\n' "$(stat -c %s "$B/out/cap-atomics.bin")" \
  "$(sha256sum "$B/out/cap-atomics.bin" | cut -c1-16)" "$("$BIN/llvm-nm" "$B/out/cap-atomics.elf" | awk '$3=="board_rec"{print $1}')"
