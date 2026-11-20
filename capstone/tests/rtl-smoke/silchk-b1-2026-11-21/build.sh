#!/bin/bash
# Silicon check list for bitstream 1 (P1 d83d3a633 + the R-76 ccsrrw term; the four fixes' files are identical to P1's):
# R-60 (item 4: an out-of-window cursor loses its tag instead of aliasing), R-68 (a superpage load is not held as
# non-idempotent), R-72 (the PC capability's bounds check covers
# the whole instruction) and R-32 (SHRINKTO gives imm bytes), as bare M-mode board images (load at 0x80000000). Each
# fix is in P1's ancestry and had simulation-only acceptance. Wrapper and env as r32-shrinkto-bare-2026-11-06 (inc/
# copied from it, identical to r58-r59-q24-bare-2026-10-26's). Sources:
#   va-runv.S          r58-r59-q24-bare-2026-10-26/tests (the control, as there)
#   cap-alias-store.S  capstone-ariane d83d3a633 verif/tests/custom/capstone
#   pcc-end-rvc.S      capstone-ariane ca1a4cdc7 (perf/r72-pccend) verif/tests/custom/capstone
#   niss-superpage.S   capstone-ariane d83d3a633, with ONE kit edit: both exits end through CAP_PASS (see the file)
#   shrinkto-readback.S and images/shrinkto.bin: r32-shrinkto-bare-2026-11-06, reused byte for byte; it is rebuilt
#                      here into out/ only to check that this compiler reproduces it.
# Recorder scratch registers from pick-regs.py (vp2-bare-2026-10-17), per test.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char regs [extra -D]
  local n=$1 f=$2 ch=$3 regs=$4 dst=${DST:-images}; shift 4
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch $regs "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/$dst/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/$dst/$n.rec"
  printf '%-18s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/$dst/$n.bin")" "$(sha256sum "$B/$dst/$n.bin" | cut -c1-16)" "$(cat "$B/$dst/$n.rec")"
}
R1="-DBOARD_R_CAP=x16 -DBOARD_R_T1=x8 -DBOARD_R_T2=x4"
R2="-DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8"
build va-runv         va-runv.S         160 "$R1"            # the control: va-runv's own arm, as in the R-58/R-59 kit
build alias-store     cap-alias-store.S 161 "$R1"            # R-60: sentinel intact and cause 24 (fixed) vs 0x9999 (alias landed)
build pcc-end         pcc-end-rvc.S     162 "$R2"            # R-72: the straddler faults at itself (fixed) vs commits
build pcc-end-in      pcc-end-rvc.S     163 "$R2" -DPE_IN    # R-72 control: wholly inside, commits on both trees
build pcc-end-out     pcc-end-rvc.S     164 "$R2" -DPE_OUT   # R-72 control: wholly outside, faults on both trees
build niss-superpage  niss-superpage.S  165 "$R2"            # R-68: per-arm cycles; arm 1 (2 MiB, vaddr[20:19]=00) vs arm 5 (4 KiB, same PA)
DST=out build shrinkto-check shrinkto-readback.S 82 "-DBOARD_R_CAP=x18 -DBOARD_R_T1=x19 -DBOARD_R_T2=x20"
cmp "$B/out/shrinkto-check.bin" "$B/images/shrinkto.bin" && echo "shrinkto reproduced byte for byte" || { echo "shrinkto NOT reproduced"; exit 1; }
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
