#!/bin/bash
# Q-04 (b) on silicon (runtime lane's request, 2026-11-03): MOVC keeps an untagged source. One bare M-mode board image
# (load at 0x80000000) of capstone-ariane movc-untagged-source.S built with -DBOARD_INTS (q04-board b534e60ca), which
# records the nine integer readings and a pass/fail marker before the checks, so either MOVC semantics returns a
# whole vector. Wrapper exactly as objb-bare-2026-10-30 (inc/ copied from it). Recorder scratch x8, x4, x2: the
# registers pick-regs.py finds unnamed by the test. Checked equal to VP_SRC when given.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
if [ -n "${VP_SRC:-}" ]; then cmp "$VP_SRC/movc-untagged-source.S" "$B/tests/movc-untagged-source.S" || { echo "tests/ differs from $VP_SRC"; exit 1; }; fi
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char [extra -D]
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch -DBOARD_R_CAP=x8 -DBOARD_R_T1=x4 -DBOARD_R_T2=x2 "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-14s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
build movc-untagged movc-untagged-source.S 160 -DBOARD_INTS
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
