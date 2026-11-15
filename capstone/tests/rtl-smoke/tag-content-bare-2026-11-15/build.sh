#!/bin/bash
# Tag content after plain overwrites, on silicon: one bare M-mode board image (load at 0x80000000) for F (a948e4134),
# F2 (eeb654104) and G (9fd9d2530). The source is capstone-ariane verif/tests/custom/capstone/tag-content.S at
# sim/tag-audit 2702f75d4 (tests only), checked equal to the tree named by VP_SRC when given. Wrapper and env as
# fp-rm-bare-2026-10-26 (inc/ copied). Recorder scratch x16, x17, x8: registers the test never names
# (vp2-bare-2026-10-17/pick-regs.py).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
if [ -n "${VP_SRC:-}" ]; then cmp "$VP_SRC/tag-content.S" "$B/tests/tag-content.S" || { echo "tests/tag-content.S differs from $VP_SRC"; exit 1; }; fi
mkdir -p "$B/out" "$B/images"
n=tag-content
"$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
  -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
  -DBOARD_TEST_FILE="\"tag-content.S\"" -DBOARD_TEST_CH=84 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8 \
  -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
"$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
"$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
printf '%-12s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
