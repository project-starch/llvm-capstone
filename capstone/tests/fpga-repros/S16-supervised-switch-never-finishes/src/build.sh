#!/bin/bash
# Rebuild this folder's image from src/sup-capstl.S with the supervised-CALL ladder's committed board harness
# (capstone/tests/rtl-smoke/sup-bare-2026-10-03-ladder). The output must match images/SHA256SUMS.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); H=$B/../../../rtl-smoke/sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
mkdir -p "$B/out"
"$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
  -fuse-ld=lld -I"$H/inc" -I"$B" -I"$H/env/macros" -I"$H/env/p" -DBOARD_TEST_FILE='"sup-capstl.S"' \
  -DBOARD_TEST_CH=122 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8 \
  -DMSWAP -DSWAP_DEP -DNOPLOOP -DTRACE_CHARS -DDOTMASK=15 -DQUANTUM=64 -DITER=65536 \
  -T "$H/inc/board.ld" "$H/inc/board_wrap.S" -o "$B/out/armdep-d16-q64.elf"
"$BIN/llvm-objcopy" -O binary "$B/out/armdep-d16-q64.elf" "$B/out/armdep-d16-q64.bin"
sha256sum "$B/out/armdep-d16-q64.bin"
# the same without the per-resume trace prints (the fastest reproduction: hangs within 16 resumes)
"$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
  -fuse-ld=lld -I"$H/inc" -I"$B" -I"$H/env/macros" -I"$H/env/p" -DBOARD_TEST_FILE='"sup-capstl.S"' \
  -DBOARD_TEST_CH=123 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8 \
  -DMSWAP -DSWAP_DEP -DNOPLOOP -DDOTMASK=15 -DQUANTUM=64 -DITER=65536 \
  -T "$H/inc/board.ld" "$H/inc/board_wrap.S" -o "$B/out/armdep-nt-d16-q64.elf"
"$BIN/llvm-objcopy" -O binary "$B/out/armdep-nt-d16-q64.elf" "$B/out/armdep-nt-d16-q64.bin"
sha256sum "$B/out/armdep-nt-d16-q64.bin"
