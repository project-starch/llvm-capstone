#!/bin/bash
# S-29's and S-23's-fetch-face arms as M-mode board images (load at 0x80000000), for reading the stale-translation
# window on SILICON. Wrapper, env and inc/ copied from vp2-bare-2026-10-17, whose 25 arms this set does NOT touch:
# that folder is live production for the tree A session's arms gate, which fails unless its table gives exactly 25
# arms, so these live here with their own id characters (130-136) and their own prefix sv-*.
#
# The source is capstone-ariane virtual-profile-2-c's verif/tests/custom/capstone/va-satpstale.S, which carries
# thirteen builds; seven are useful on a board. Eight of vp2's existing bare arms already set satp and four already
# issue CSRUNV, so translation and virtual contexts are not a new class of arm here.
#
# THE BOARD VECTOR IS NOT THE SIMULATION VECTOR. The recorder's CAPPRINT creates its own capability, so node counts
# differ, and the board's wrapper page puts _start one page above RAM_BASE. Every vector for these images comes from
# simulating THESE images, on tree A's RTL (8522d72fe, resident) and on tree B's (8de3c48cf), never from the arm's
# own simulation readings.
#
# Register picks: pick-regs.py returns x16/x8/x4 for this source, the same three the va-runv family uses, so the
# arm names none of them.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
if [ -n "${VP_SRC:-}" ]; then
  cmp "$VP_SRC/va-satpstale.S" "$B/tests/va-satpstale.S" || { echo "tests/va-satpstale.S differs from $VP_SRC"; exit 1; }
fi
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char regs [extra -D]
  local n=$1 f=$2 ch=$3 r=$4; shift 4
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$B/tests/inc" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch $r "$@" -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-16s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
R="-DBOARD_R_CAP=x16 -DBOARD_R_T1=x8 -DBOARD_R_T2=x4"

# The DATA face, four arms differing only in where the caller fences. On tree A the unfenced pair should read the
# stale values; on tree B all four should read clean.
build sv-stale      va-satpstale.S 130 "$R"
build sv-stale-f    va-satpstale.S 131 "$R" -DFENCE_CTRL
build sv-stale-b    va-satpstale.S 132 "$R" -DFENCE_BEFORE
build sv-stale-bf   va-satpstale.S 133 "$R" -DFENCE_BEFORE -DFENCE_CTRL
# The control that attributes the escape-direction reading to the context's own access.
build sv-stale-bn   va-satpstale.S 134 "$R" -DFENCE_BEFORE -DNO_CTX_READ
# The FETCH face: the caller drops to U for its first fetch only, so a stale entry reaches the capability check and
# is refused with a PHYSICAL address, which is S-23's signature.
build sv-fetch      va-satpstale.S 135 "$R" -DFENCE_BEFORE -DFETCH_FACE -DFETCH_GAP -DCALLER_U
build sv-fetch-c    va-satpstale.S 136 "$R" -DFENCE_BEFORE -DFETCH_FACE -DFETCH_GAP -DCALLER_U -DFENCE_CTRL

(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
(cd "$B/out" && sha256sum *.elf > SHA256SUMS)
