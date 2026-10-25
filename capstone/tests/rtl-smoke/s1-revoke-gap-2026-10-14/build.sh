#!/bin/bash
# Build the s1-revoke-gap pair as bare M-mode board images (load at 0x80000000), the same binary for the FP and the
# Sublet bitstream. Wrapper, linker script, env headers: ../sup-bare-2026-10-03-ladder (unchanged). The test source is
# capstone-ariane verif/tests/custom/capstone/s1-revoke-gap.S, copied here at build time so the folder is self-contained;
# the copy must equal the RTL tree's file (the build checks).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder   # env/ (riscv-tests headers) only; inc/ is this folder's node-aware copy
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
SRC=${S1GAP_SRC:-$B/../../../capstone-ariane/verif/tests/custom/capstone/s1-revoke-gap.S}
mkdir -p "$B/out" "$B/images" "$B/tests"; cp "$SRC" "$B/tests/s1-revoke-gap.S"
build() {   # name id-char [extra -D]
  local n=$1 ch=$2; shift 2
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"s1-revoke-gap.S\"" -DBOARD_TEST_CH=$ch "$@" -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8 -DBOARD_REC_NODE=x18 \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-14s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
build s1gap-close 67 -DGAP=0
build s1gap-far   70 -DGAP=1000
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
