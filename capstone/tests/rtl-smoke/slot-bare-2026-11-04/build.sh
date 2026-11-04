#!/bin/bash
# STC/LDC pair cost on silicon (SPEC lane's request, 2026-11-04): bare M-mode board images (load at 0x80000000) of
# capstone-ariane stc-ldc-slot.S, the same blob (868111d4cd6e) on c5a-eval-ctl 3afd5c9e2 (bitstream 3's tokens), objb-eval
# d463206ac (ObjBounds 603528b8b + the test) and c5a-eval 08c9fbd22 (C.5a 1fc2cd5aa + the test). Wrapper exactly as
# hpm-bare-2026-11-03 (inc/ copied from it). Recorder scratch x16, x17, x4: pick-regs.py's choice for this test.
# Checked equal to VP_SRC when given.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
if [ -n "${VP_SRC:-}" ]; then cmp "$VP_SRC/stc-ldc-slot.S" "$B/tests/stc-ldc-slot.S" || { echo "tests/stc-ldc-slot.S differs from $VP_SRC"; exit 1; }; fi
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char [extra -D]
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x4 "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-8s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
# Every arm returns on every bitstream; any order. Recommended: 9 (the bare loop, the control) first.
build slot-9 stc-ldc-slot.S 169 -DSL=9   # the loop alone: control
build slot-6 stc-ldc-slot.S 166 -DSL=6   # sd A; ld A+0x10
build slot-5 stc-ldc-slot.S 165 -DSL=5   # sd A; ld A
build slot-7 stc-ldc-slot.S 167 -DSL=7   # ldc A alone
build slot-8 stc-ldc-slot.S 168 -DSL=8   # stc A alone
build slot-1 stc-ldc-slot.S 161 -DSL=1   # stc A; ldc A
build slot-2 stc-ldc-slot.S 162 -DSL=2   # stc A; ldc A+0x10
build slot-3 stc-ldc-slot.S 163 -DSL=3   # stc A; ldc A+0x40
build slot-4 stc-ldc-slot.S 164 -DSL=4   # stc A; ldc A+0x1000 (same page offset)
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
