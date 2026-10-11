#!/bin/bash
# Private QEMU fw_jump: <out-dir> <wrapper-commit> <monitor-commit-or-ref> [extra -D defines...]
# Recipe from memory note process-abi-needs-private-firmware: git archive, capstone-c .c.S, make PLATFORM=generic.
set -euo pipefail
OUT=$1 WREV=$2 MREV=$3; shift 3
WRAP=$HOME/dev/llvm-capstone/capstone/caplifive-buildroot/components/opensbi
MON=$HOME/dev/llvm-capstone/capstone/caplifive-buildroot/components/opensbi/lib/sbi/capstone-sbi
CC=$HOME/dev/llvm-capstone/capstone/capstone-c/target/debug/capstone-c
XC=$HOME/dev/llvm-capstone/capstone/caplifive-buildroot/build-qemu/host/bin/riscv64-buildroot-linux-gnu-
rm -rf "$OUT"; mkdir -p "$OUT/opensbi"
git -C "$WRAP" archive "$WREV" | tar -x -C "$OUT/opensbi"
rm -rf "$OUT/opensbi/lib/sbi/capstone-sbi"; mkdir -p "$OUT/opensbi/lib/sbi/capstone-sbi"
git -C "$MON" archive "$MREV" | tar -x -C "$OUT/opensbi/lib/sbi/capstone-sbi"
DEFS="-DCAPSTONE_TARGET_QEMU -DCAPSTONE_DEBUG_ENABLE $*"
for f in lib/sbi/sbi_capstone_dom.c lib/sbi/capstone_int_handler.c; do
  "$CC" --abi capstone "$OUT/opensbi/$f" -- -I"$OUT/opensbi/lib/sbi/capstone-sbi" -D__riscv_xlen=64 $DEFS > "$OUT/opensbi/$f.S" 2> "$OUT/$(basename $f).capstone-c.log"
done
taskset -c 0-7,32-39 nice -n 10 make -s -C "$OUT/opensbi" PLATFORM=generic CROSS_COMPILE="$XC" -j12 > "$OUT/make.log" 2>&1
ls -la "$OUT/opensbi/build/platform/generic/firmware/fw_jump.elf"
echo "defs: $DEFS"; echo "capstone-c: $(git -C $HOME/dev/llvm-capstone/capstone/capstone-c rev-parse --short HEAD)"
