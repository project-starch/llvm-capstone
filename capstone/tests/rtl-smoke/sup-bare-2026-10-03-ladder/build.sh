#!/bin/bash
# Build the three supervised-CALL directed tests as bare M-mode board images (load at 0x80000000).
# Self-contained: env/ holds the riscv-tests headers the run was built with (capstone-ariane's riscv-tests
# checkout, env 2f75dc2 plus local modifications to p/riscv_test.h). Only the compiler is external.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd)
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
mkdir -p "$B/out"
build() {   # name test-file id-char [extra -D]
  local n=$1 f=$2 ch=$3; shift 3
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$B/env/macros" -I"$B/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch "$@" -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/out/$n.bin"
  printf '%-14s %8d bytes  sha256 %s\n' "$n" "$(stat -c %s "$B/out/$n.bin")" "$(sha256sum "$B/out/$n.bin" | cut -c1-16)"
}
build sup-fullswitch       sup-fullswitch.S 70  -DBOARD_R_CAP=x16 -DBOARD_R_T1=x8 -DBOARD_R_T2=x9
build sup-strip            sup-strip.S      83  -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8
build sup-strip-d0         sup-strip.S      48 -DSTRIP_DELAY=0 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8
build sup-strip-d8         sup-strip.S      56 -DSTRIP_DELAY=8 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8
build sup-strip-d32        sup-strip.S      51 -DSTRIP_DELAY=32 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8
build sup-strip-d128       sup-strip.S      49 -DSTRIP_DELAY=128 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8
build sup-arm              sup-arm.S        65  -DBOARD_R_CAP=x8 -DBOARD_R_T1=x29 -DBOARD_R_T2=x30
build sup-armclose         sup-armclose.S   67  -DBOARD_R_CAP=x17 -DBOARD_R_T1=x8 -DBOARD_R_T2=x7
build sup-sealsize-64      sup-sealsize.S   54 -DSEAL_SIZE=64 -DBOARD_R_CAP=x17 -DBOARD_R_T1=x8 -DBOARD_R_T2=x7
build sup-sealsize-1023    sup-sealsize.S   50 -DSEAL_SIZE=1023 -DBOARD_R_CAP=x17 -DBOARD_R_T1=x8 -DBOARD_R_T2=x7
build sup-sealsize-1024    sup-sealsize.S   52 -DSEAL_SIZE=1024 -DBOARD_R_CAP=x17 -DBOARD_R_T1=x8 -DBOARD_R_T2=x7
build sup-sealsize-2048    sup-sealsize.S   53 -DSEAL_SIZE=2048 -DBOARD_R_CAP=x17 -DBOARD_R_T1=x8 -DBOARD_R_T2=x7
build sup-sealsize-off8    sup-sealsize.S   57 -DSEAL_SIZE=1024 -DSEAL_OFF=8 -DBOARD_R_CAP=x17 -DBOARD_R_T1=x8 -DBOARD_R_T2=x7
build sup-sealsize-off16   sup-sealsize.S   55 -DSEAL_SIZE=1024 -DSEAL_OFF=16 -DBOARD_R_CAP=x17 -DBOARD_R_T1=x8 -DBOARD_R_T2=x7
build sup-sealsize-cur960  sup-sealsize.S   57 -DSEAL_SIZE=1024 -DSEAL_CURSOR=960 -DBOARD_R_CAP=x17 -DBOARD_R_T1=x8 -DBOARD_R_T2=x7
build sup-guards           sup-guards.S     71  -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8
build sup-mtip             sup-mtip.S       77  -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8
