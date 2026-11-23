#!/bin/bash
# R-79 (the PMP judged an access with the NEIGHBOURING cycle's access type) as a bare M-mode board image (load at
# 0x80000000), for any bitstream: plain RISC-V only, no capability instruction runs in S. Wrapper and env exactly as
# r61-stale-bare-2026-10-24 (inc/ copied from it). Recorder scratch x8, x4, x2: unnamed by the test (its header lists
# every register it writes). The source derives from capstone-ariane b5b1-step3 csmint-pmp.S, without the CSMINT arms.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char [extra -D]
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch -DBOARD_R_CAP=x8 -DBOARD_R_T1=x4 -DBOARD_R_T2=x2 "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-16s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
build r79-pmp r79-pmp.S 82
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
