#!/bin/bash
# R-32 (SHRINKTO's end) on silicon: one bare M-mode board image (load at 0x80000000). The test is written for this kit
# (tests/shrinkto-readback.S); the wrapper is fp-rm-bare-2026-10-26's (inc/ copied unchanged). Recorder scratch
# x18, x19, x20: registers the test never names (CAP_PASS uses x23-x31).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
mkdir -p "$B/out" "$B/images"
n=shrinkto
"$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
  -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
  -DBOARD_TEST_FILE="\"shrinkto-readback.S\"" -DBOARD_TEST_CH=82 -DBOARD_R_CAP=x18 -DBOARD_R_T1=x19 -DBOARD_R_T2=x20 \
  -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
"$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
"$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
printf '%-10s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
