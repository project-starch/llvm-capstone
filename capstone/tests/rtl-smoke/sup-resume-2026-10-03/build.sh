#!/bin/bash
# Build the sup-capstl variants as bare M-mode board images (load at 0x80000000), with the supervised-CALL ladder's
# committed board harness (../sup-bare-2026-10-03-ladder: inc/ + env/).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd)
H=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
mkdir -p "$B/out"
build() {   # name id-char [extra -D]
  local n=$1 ch=$2; shift 2
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$H/inc" -I"$B/tests" -I"$H/env/macros" -I"$H/env/p" \
    -DBOARD_TEST_FILE='"sup-capstl.S"' -DBOARD_TEST_CH=$ch -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8 "$@" \
    -T "$H/inc/board.ld" "$H/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/out/$n.bin"
  printf '%-14s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/out/$n.bin")" \
    "$(sha256sum "$B/out/$n.bin" | cut -c1-16)" "$("$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}')"
}
build intloop-q64 73 -DINTLOOP -DQUANTUM=64
build capstl-q64  67 -DQUANTUM=64
build capstl-q47  68 -DQUANTUM=47
build capstl-q16  69 -DQUANTUM=16
# cache-miss variants (after the hot-cache session completed every arm): ITER 2^20 for the domain-side sweep
build evict-noploop-q100k 78 -DEVICT -DNOPLOOP -DQUANTUM=100000 -DITER=1048576
build evict-intloop-q100k 79 -DEVICT -DINTLOOP -DQUANTUM=100000 -DITER=1048576
build evict-capstl-q100k  80 -DEVICT -DQUANTUM=100000 -DITER=1048576
build evict-capstl-q20k   81 -DEVICT -DQUANTUM=20000 -DITER=1048576
build mevict-noploop-q64  82 -DMEVICT -DNOPLOOP -DQUANTUM=64
build mevict-capstl-q64   83 -DMEVICT -DQUANTUM=64
# the FPGA monitor's __domcallsaves sequence around every CALL (CPMP/CSR swap, cscratch := sp, sp := 0)
build mswap-capstl-q64          84 -DMSWAP -DQUANTUM=64
build mswap-evict-noploop-q100k 85 -DMSWAP -DEVICT -DNOPLOOP -DQUANTUM=100000 -DITER=1048576
build mswap-evict-capstl-q100k  86 -DMSWAP -DEVICT -DQUANTUM=100000 -DITER=1048576
build mswap-mevict-capstl-q64   87 -DMSWAP -DMEVICT -DQUANTUM=64
