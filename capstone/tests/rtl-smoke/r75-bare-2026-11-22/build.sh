#!/bin/bash
# R-75 silicon reproducer for bitstream 1 (9e801ab5e) against R-75 option (d) (capstone-ariane b1-r75d 1c5d31474, sealed
# 2d2ba92c2e19...): bare M-mode board images (load at 0x80000000), the silchk-b1-2026-11-21 kit's wrapper and env (inc/
# copied from it). Sources:
#   va-runv.S        silchk-b1-2026-11-21/tests (the control, known silicon vector)
#   va-fetchsplit.S  capstone-ariane 1c5d31474 verif/tests/custom/capstone (R-75's arms are its -DVA_RC_* variants)
# Recorder scratch registers: pick-regs.py (vp2-bare-2026-10-17) run on the lines ACTIVE under each variant's defines
# (va-fetchsplit.S names every GPR across all its variants), by an #if evaluator kept beside this script.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char regs [extra -D]
  local n=$1 f=$2 ch=$3 regs=$4; shift 4
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch $regs "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-14s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
RC="-DVA_RC_CTX -DVA_RC_TRAP -DFS_SAMETBL -DCS4_TWO"
build va-runv     va-runv.S       160 "-DBOARD_R_CAP=x16 -DBOARD_R_T1=x8 -DBOARD_R_T2=x4"               # the control
build r75-trapz   va-fetchsplit.S 170 "-DBOARD_R_CAP=x4 -DBOARD_R_T1=x2 -DBOARD_R_T2=x31" $RC -DCS4_ZERO  # same scaffold, returns on both trees
build r75-trap    va-fetchsplit.S 171 "-DBOARD_R_CAP=x4 -DBOARD_R_T1=x2 -DBOARD_R_T2=x30" $RC             # R-75: loops on cause 25 on bitstream 1
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
