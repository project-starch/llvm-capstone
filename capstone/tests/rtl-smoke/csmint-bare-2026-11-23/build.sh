#!/bin/bash
# B5b-1 step 3 on silicon: CSMINT (and the Sublet regression) as bare M-mode board images (load at 0x80000000), for the
# first B5b-1 bitstream (capstone-ariane b5b1-step3 d0afdb40f). Wrapper and env exactly as r79-pmp-bare-2026-11-23
# (inc/, also r61-stale-bare's). The tests are capstone-ariane verif/tests/custom/capstone/{csmint,csmint-vm,csmint-pmp,
# sublet-life}.S at d0afdb40f, unchanged (checked equal to the tree named by VP_SRC when given). Recorder scratch x8, x4,
# x2: none is read or written by any of the four (csmint.S's a15 ENCODES rs1 = x2 and traps at decode).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
if [ -n "${VP_SRC:-}" ]; then for f in csmint.S csmint-vm.S csmint-pmp.S sublet-life.S; do cmp "$VP_SRC/$f" "$B/tests/$f" || { echo "tests/$f differs from $VP_SRC"; exit 1; }; done; fi
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char [extra -D]
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch -DBOARD_R_CAP=x8 -DBOARD_R_T1=x4 -DBOARD_R_T2=x2 "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-18s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
build sublet-life      sublet-life.S 83                 # the Sublet regression control (step 2a's pool arms)
build csmint           csmint.S      84                 # 18 arms on the physical pool, integer M
build csmint-vm        csmint-vm.S   85                 # S under Sv39: the descriptor's and the slots' page faults
build csmint-vm-nc     csmint-vm.S   86 -DCM_NONCANON   # ... with a non-canonical descriptor and slot
build csmint-pmp       csmint-pmp.S  87                 # R-79's PMP arms and CSMINT's probe against an R-only page
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
