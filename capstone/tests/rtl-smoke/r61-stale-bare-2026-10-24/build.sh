#!/bin/bash
# R-61 (a refused consuming LDC leaves the load unit's store-permission flag at 0, and a later LEGAL first-touch load in a
# virtual context faults with STORE_PAGE_FAULT) as bare M-mode board images (load at 0x80000000) for the batch bitstream
# c721c1a2a. Wrapper, env and the node-aware header exactly as va-alloc-bare-2026-10-23 (inc/ copied from it; the wrapper
# is also vp2-bare-2026-10-17's). The source is a copy of capstone-ariane verif/tests/custom/capstone/vb-trans.S at
# vp3-ringcut dcc6daf6e (tests only above the RTL under test), checked equal to the tree named by VP_SRC at build time.
# Recorder scratch x8, x4, x2: the registers pick-regs.py finds unnamed by the test, the same set vp2-bare used for
# vb-trans. All three images are VB_TABLE builds (a domain under csupctl.virt needs a lifetime table since item A).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
if [ -n "${VP_SRC:-}" ]; then cmp "$VP_SRC/vb-trans.S" "$B/tests/vb-trans.S" || { echo "tests/vb-trans.S differs from $VP_SRC"; exit 1; }; fi
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char [extra -D]
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch -DBOARD_R_CAP=x8 -DBOARD_R_T1=x4 -DBOARD_R_T2=x2 -DVB_TABLE "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-16s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
build r61-control vb-trans.S 130                # the full vp2 arm list: the known-good control for this boot
build r61-stale2  vb-trans.S 131 -DVB_STALE2    # a refused consuming ldc, then a LEGAL first-touch load: the R-61 reading
build r61-stale   vb-trans.S 132 -DVB_STALE     # a refused consuming ldc, then a load from an unmapped page
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
