#!/bin/bash
# R-67 bare board kit: conformance rule T1 (a cjalr through a code capability whose node has ended) as four bare M-mode
# images (load at 0x80000000): FIRE and its three controls (norevoke, untagged, oob). The test sources are the external
# collaborator's (capstone-ariane perf/c5a-cached-ldc fe5aa8440, verif/tests/custom/capstone/lv-t1-*), unchanged; the
# wrapper is objb-bare-2026-10-30's (inc/ copied from it). Recorder scratch x16, x8, x4: the registers pick-regs.py
# finds unnamed by the test. Checked equal to VP_SRC when given.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../../rtl-smoke/sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
if [ -n "${VP_SRC:-}" ]; then
  for f in lv-t1-body.h lv-t1-fire.S lv-t1-norevoke.S lv-t1-untagged.S lv-t1-oob.S; do cmp "$VP_SRC/$f" "$B/tests/$f" || { echo "tests/$f differs from $VP_SRC"; exit 1; }; done
fi
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch -DBOARD_R_CAP=x16 -DBOARD_R_T1=x8 -DBOARD_R_T2=x4 "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-14s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
# Boot order: the three controls first (each returns on every bitstream), FIRE last.
build t1-norevoke lv-t1-norevoke.S 170
build t1-untagged lv-t1-untagged.S 171
build t1-oob      lv-t1-oob.S      172
build t1-fire     lv-t1-fire.S     173
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
