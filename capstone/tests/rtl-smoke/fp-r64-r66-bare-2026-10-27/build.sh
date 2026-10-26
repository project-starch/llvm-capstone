#!/bin/bash
# R-64, R-65 and R-66 (the divide/sqrt unit's flags, overflow results, subnormal-to-normal carry and NaN boxing) on
# silicon: five bare M-mode board images (load at 0x80000000) for bitstream 3 (caplifive_vp2c_e7af48457), which carries
# none of their fixes. Sources are capstone-ariane verif/tests/custom/capstone/<name>-divsqrt.S at r62-divsqrt be9a844b7
# (tests only), checked equal to the tree named by VP_SRC when given. Wrapper and env as fp-rm-bare-2026-10-26 (inc/
# copied). Recorder scratch registers per image from vp2-bare-2026-10-17/pick-regs.py (registers the test never names).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
mkdir -p "$B/out" "$B/images"
for spec in "fp-of x16 x17 x4" "fp-special x16 x17 x8" "fp-carry x16 x17 x4" "fp-box x16 x17 x8" "fp-edge x16 x17 x4"; do
  set -- $spec; n=$1
  if [ -n "${VP_SRC:-}" ]; then cmp "$VP_SRC/$n-divsqrt.S" "$B/tests/$n-divsqrt.S" || { echo "tests/$n-divsqrt.S differs from $VP_SRC"; exit 1; }; fi
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$n-divsqrt.S\"" -DBOARD_TEST_CH=163 -DBOARD_R_CAP=$2 -DBOARD_R_T1=$3 -DBOARD_R_T2=$4 \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-11s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
done
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
