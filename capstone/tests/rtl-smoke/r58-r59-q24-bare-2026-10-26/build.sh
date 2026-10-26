#!/bin/bash
# R-58 (the CSRUNV caller's srevroot restored at the escape), R-59 (the precheck refuses a table word with low bits
# set) and Q-24 (firmware may delegate a capability cause to S) as bare M-mode board images (load at 0x80000000) for bitstream 2 (caplifive_vp2c_dcc6daf6e). Both fixes are in
# RTL since the batch c721c1a2a and had simulation-only acceptance. Wrapper, env and node-aware header as
# r61-stale-bare-2026-10-24 (inc/ copied from it; vp2-bare-2026-10-17's wrapper). Sources are copies of capstone-ariane
# verif/tests/custom/capstone/{va-runv,va-srevroot}.S at dcc6daf6e and q24-deleg.S at vp3-ringcut f68882f1e (tests only above dcc6daf6e), checked equal to the tree named by VP_SRC when given.
# Recorder scratch x16, x8, x4: registers none of the three tests names (pick-regs.py; vp2-bare's set for va-runv).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
if [ -n "${VP_SRC:-}" ]; then for f in va-runv.S va-srevroot.S q24-deleg.S; do cmp "$VP_SRC/$f" "$B/tests/$f" || { echo "tests/$f differs from $VP_SRC"; exit 1; }; done; fi
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char [extra -D]
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch -DBOARD_R_CAP=x16 -DBOARD_R_T1=x8 -DBOARD_R_T2=x4 "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-18s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
build va-runv          va-runv.S     150                 # the control: va-runv's own arm (and the base of the two below)
build va-srevroot      va-srevroot.S 151                 # R-58: the caller reads its OWN srevroot back after the escape
build va-runv-r59      va-runv.S     152 -DVA_R59        # R-59: tbl with bit 11 set, refused at entry
build va-runv-r59-low  va-runv.S     153 -DVA_R59_LOW    # R-59: the word 0x800 alone, refused at entry
build q24-deleg        q24-deleg.S   154                 # Q-24 control: medeleg clear, the 24 stays in M
build q24-deleg-on     q24-deleg.S   155 -DQ24_DELEG     # Q-24: medeleg bit 24 set, the 24 lands in S
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
