#!/bin/bash
# The Sublet bare arms as M-mode board images (load at 0x80000000): the acceptance floor for caplifive_sublet_*.bit and
# the negative control on the FP bitstream (the first CDERIVE must trap 2). Wrapper and env as sup-bare-2026-10-03-ladder;
# inc/ is the node-aware header (CAPPRINT / CAP_PASS attach their record capabilities to a test-chosen node, needed by
# the arm that revokes the base's node). Sources are copies of capstone-ariane verif/tests/custom/capstone, checked equal.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
SRC=${SUBLET_SRC:-$B/../../../capstone-ariane/verif/tests/custom/capstone}
for f in sublet-derive-kill.S sublet-split-shares-node.S sublet-mrev-orphan.S; do cmp "$SRC/$f" "$B/tests/$f" || { echo "tests/$f differs from the RTL tree"; exit 1; }; done
build() {   # name test-file id-char [extra -D]
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch "$@" -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-26s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
# registers the tests never name: derive-kill and split-shares-node leave x8/x16/x17 free and never revoke the base's node
build sublet-derive-kill       sublet-derive-kill.S       68 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8
build sublet-split-shares-node sublet-split-shares-node.S 83 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8
# mrev-orphan uses a6; its revokes end the base's node, so the record attaches to the tohost sibling's node (s11 = x27)
build sublet-mrev-orphan       sublet-mrev-orphan.S       79 -DBOARD_R_CAP=x9 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8 -DBOARD_REC_NODE=x27
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
