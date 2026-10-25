#!/bin/bash
# s23-cpmp-rearm as a bare M-mode board image (load at 0x80000000): the must-read arm of the batch bitstream's first
# boot. Wrapper, env and the node-aware header exactly as vp1-bare-2026-10-17 (inc/ copied from it). The source is a
# copy of capstone-ariane verif/tests/custom/capstone/s23-cpmp-rearm.S at the batch tip c721c1a2a, checked equal to
# the tree named by VP_SRC at build time. The arm leaves x8, x4 and x2 untouched (it names none of s0/sp/tp), so the
# wrapper's three scratch registers take them, as vb-trans did. No stack is used.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
SRC=${VP_SRC:-$B/../../../capstone-ariane/verif/tests/custom/capstone}
for f in s23-cpmp-rearm.S; do cmp "$SRC/$f" "$B/tests/$f" || { echo "tests/$f differs from $SRC"; exit 1; }; done
build() {   # name test-file id-char [extra -D]
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch "$@" -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-16s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
build s23-cpmp-rearm s23-cpmp-rearm.S 90 -DBOARD_R_CAP=x8 -DBOARD_R_T1=x4 -DBOARD_R_T2=x2
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
