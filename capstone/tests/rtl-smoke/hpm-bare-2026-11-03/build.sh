#!/bin/bash
# Stock CVA6 performance counters on silicon (SPEC lane's request, 2026-11-03): bare M-mode board images (load at
# 0x80000000) of capstone-ariane hpm-probe.S, the same source on hpm-probe-b3 87368cc69 (bitstream 3's tokens) and
# hpm-probe-objb fc1114913 (ObjBounds 603528b8b + the test). Wrapper exactly as objb-bare-2026-10-30 (inc/ copied from
# it). Recorder scratch x4, x2, x9: the registers pick-regs.py finds unnamed by the test. Checked equal to VP_SRC when given.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
if [ -n "${VP_SRC:-}" ]; then cmp "$VP_SRC/hpm-probe.S" "$B/tests/hpm-probe.S" || { echo "tests/hpm-probe.S differs from $VP_SRC"; exit 1; }; fi
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char [extra -D]
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch -DBOARD_R_CAP=x4 -DBOARD_R_T1=x2 -DBOARD_R_T2=x9 "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-12s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
# Every arm returns on both bitstreams; any order. Recommended: 5 (empty, the control) first.
build hpm-5 hpm-probe.S 155 -DHPM=5   # empty loop: control
build hpm-1 hpm-probe.S 151 -DHPM=1   # ld
build hpm-2 hpm-probe.S 152 -DHPM=2   # sd
build hpm-3 hpm-probe.S 153 -DHPM=3   # ldc
build hpm-4 hpm-probe.S 154 -DHPM=4   # stc
build hpm-6 hpm-probe.S 156 -DHPM=6   # ld with mcountinhibit[8:3] set
build hpm-7 hpm-probe.S 157 -DHPM=7   # ld, 16 B stride over a 4 KiB region
build hpm-8 hpm-probe.S 158 -DHPM=8   # L1I-miss positive control (2026-11-04)
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
