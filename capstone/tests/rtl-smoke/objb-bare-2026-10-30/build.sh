#!/bin/bash
# ObjBounds (CAPSTONE_NO_LIFETIME_CHECK, capstone-ariane objbounds-b 603528b8b): bare M-mode board images (load at
# 0x80000000) that grade the variant's identity by value against bitstream 3 (e7af48457) and read the per-access
# lifetime check's cost shape on silicon. Wrapper, env and node-aware header exactly as va-alloc-bare-2026-10-23
# (inc/ copied from it). The three sources are capstone-ariane verif/tests/custom/capstone/ at 0af4307fc, where they
# were built; they are identical at objbounds-b 603528b8b and at its control objbounds-off-b 1782d20bd. They are checked
# equal to the tree named by VP_SRC at build time. Recorder scratch x16, x17, x8: the registers pick-regs.py finds unnamed by every test here.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
if [ -n "${VP_SRC:-}" ]; then
  for f in stale-access.S objb-perf.S objb-pcc.S; do cmp "$VP_SRC/$f" "$B/tests/$f" || { echo "tests/$f differs from $VP_SRC"; exit 1; }; done
fi
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char [extra -D]
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8 "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-16s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
# Boot order is the runner's; the recommended order is this one: equal-on-both-trees control, identity, cost shapes,
# and objb-pcc LAST (on bitstream 3 it does not return: a revoked PC capability faults in its own trap handler).
build objb-pcc-ctrl objb-pcc.S   140 -DPCCTRL     # control: the same stream revoking a DATA node; equal on both trees
build objb-stale    stale-access.S 141            # identity: 25 x6 on bitstream 3, 0 x6 and the stale store landed on ObjBounds
build objb-perf-s1  objb-perf.S  142 -DSHAPE=1    # one capability, 4,096 loads
build objb-perf-s2  objb-perf.S  143 -DSHAPE=2    # chase over 64 nodes
build objb-perf-s3  objb-perf.S  144 -DSHAPE=3    # chase over 1,024 nodes
build objb-perf-s4  objb-perf.S  145 -DSHAPE=4    # 64 nodes after a namespace flush
build objb-perf-s5  objb-perf.S  146 -DSHAPE=5    # ObjLife-style churn over 1,024 objects
build objb-perf-s6  objb-perf.S  148 -DSHAPE=6    # 1,024 ldc through one capability
build objb-perf-s7  objb-perf.S  149 -DSHAPE=7    # 1,024 stc through one capability
build objb-pcc      objb-pcc.S   147              # fetch face: completes on ObjBounds only (LAST)
build objb-stale-w  objb-stale-w.S 150           # addendum 2 witness: stale-access + the DYN TIGHTEN query on each node (its own boot)
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
