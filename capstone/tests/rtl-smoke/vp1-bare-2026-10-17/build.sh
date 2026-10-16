#!/bin/bash
# The vp1 bare arms as M-mode board images (load at 0x80000000): item B's translation under a supervised domain and Q-17's
# check order, the first thing to run on caplifive_vp1_<commit>.bit after the physical regression. Wrapper, env and the
# node-aware header as sublet-bare-2026-10-15 (inc/ copied from it). Sources are copies of capstone-ariane
# verif/tests/custom/capstone at virtual-profile 9ed9a2594, checked equal to the tree named by VP_SRC at build time.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
SRC=${VP_SRC:-$B/../../../capstone-ariane/verif/tests/custom/capstone}
for f in vb-trans.S q17-order.S; do cmp "$SRC/$f" "$B/tests/$f" || { echo "tests/$f differs from $SRC"; exit 1; }; done
build() {   # name test-file id-char [extra -D]
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch "$@" -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-16s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
# vb-trans leaves exactly x2, x4 and x8 free (its monitor uses every other GPR); the wrapper's three scratch registers
# take them: no stack is used by the test or the wrapper. The in-domain CAPPRINTs compile out under BOARD_TEST_CH.
build vb-trans     vb-trans.S  86 -DBOARD_R_CAP=x8 -DBOARD_R_T1=x4 -DBOARD_R_T2=x2
build vb-trans-off vb-trans.S  87 -DBOARD_R_CAP=x8 -DBOARD_R_T1=x4 -DBOARD_R_T2=x2 -DVB_FLAG_OFF
# q17-order leaves x8/x16/x17 free
build q17-order    q17-order.S 88 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
