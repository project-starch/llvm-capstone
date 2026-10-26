#!/bin/bash
# R-63 (fdiv/fsqrt under RDN, RUP and RMM) on silicon: one bare M-mode board image (load at 0x80000000) for bitstream 2
# (caplifive_vp2c_dcc6daf6e). The source is capstone-ariane verif/tests/custom/capstone/fp-rm-divsqrt.S at r62-divsqrt
# e7af48457 (tests only), checked equal to the tree named by VP_SRC when given. Wrapper and env as
# fp-divsqrt-bare-2026-10-26 (inc/ copied). Recorder scratch x16, x17, x4: registers the test never names
# (vp2-bare-2026-10-17/pick-regs.py).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
if [ -n "${VP_SRC:-}" ]; then cmp "$VP_SRC/fp-rm-divsqrt.S" "$B/tests/fp-rm-divsqrt.S" || { echo "tests/fp-rm-divsqrt.S differs from $VP_SRC"; exit 1; }; fi
mkdir -p "$B/out" "$B/images"
n=fp-rm
"$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
  -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
  -DBOARD_TEST_FILE="\"fp-rm-divsqrt.S\"" -DBOARD_TEST_CH=163 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x4 \
  -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
"$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
"$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
printf '%-10s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
