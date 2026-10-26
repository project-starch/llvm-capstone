#!/bin/bash
# fdiv.d / fsqrt.d rounding on silicon: bare M-mode board images (load at 0x80000000) for bitstream 2
# (caplifive_vp2c_dcc6daf6e). The source is capstone-ariane verif/tests/custom/capstone/fp-ieee-divsqrt.S (vp3-ringcut,
# tests only above dcc6daf6e; checked equal to the tree named by VP_SRC when given); its 512-row table comes from
# tests/gen-fp-ieee-divsqrt.py (seed 20261026; the -DONECASE block was added by hand afterwards and is the only difference).
# Wrapper and env as r58-r59-q24-bare-2026-10-26 (inc/ copied). Recorder scratch x17, x4, x2: registers the test never
# names (vp2-bare-2026-10-17/pick-regs.py).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
if [ -n "${VP_SRC:-}" ]; then cmp "$VP_SRC/fp-ieee-divsqrt.S" "$B/tests/fp-ieee-divsqrt.S" || { echo "tests/fp-ieee-divsqrt.S differs from $VP_SRC"; exit 1; }; fi
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char [extra -D]
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch -DBOARD_R_CAP=x17 -DBOARD_R_T1=x4 -DBOARD_R_T2=x2 "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-18s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
build fp-c3    fp-ieee-divsqrt.S 160 -DONECASE=3    # case 3 alone: the fdiv rounds to the wrong neighbour in sim
build fp-c15   fp-ieee-divsqrt.S 161 -DONECASE=15   # case 15 alone: the fsqrt
build fp-all   fp-ieee-divsqrt.S 162                # all 512 cases: 23 fdiv and 19 fsqrt misses in sim
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
