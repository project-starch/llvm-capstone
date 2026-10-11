#!/bin/bash
# Rebuild this folder's three images from src/sup-capstl.S with the supervised-CALL ladder's committed board harness
# (capstone/tests/rtl-smoke/sup-bare-2026-10-03-ladder). The outputs must match images/SHA256SUMS.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); H=$B/../../../rtl-smoke/sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
mkdir -p "$B/out"
b() { local n=$1 ch=$2; shift 2
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$H/inc" -I"$B" -I"$H/env/macros" -I"$H/env/p" -DBOARD_TEST_FILE='"sup-capstl.S"' \
    -DBOARD_TEST_CH=$ch -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8 "$@" \
    -T "$H/inc/board.ld" "$H/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/out/$n.bin"; sha256sum "$B/out/$n.bin"; }
b arm12-ldc-q64 120 -DMSWAP -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=12 -DQUANTUM=64 -DITER=65536
b arm12-ld-q64  121 -DMSWAP -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=12 -DLDTEST -DQUANTUM=64 -DITER=65536
b mswapfix-plain-noploop 94 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DQUANTUM=64
