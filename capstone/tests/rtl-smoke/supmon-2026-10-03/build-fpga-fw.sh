#!/bin/bash
# Private FPGA fw_payload: <out-dir> "<defines>" -- wrapper wrapper/supcall-fpga (849c8e1) + monitor/supcall-fpga
# (1fd1bbe), payload = the shared board build's CURRENT images/Image + caplifive.dtb (copied, never modified).
set -euo pipefail
OUT=$1; DEFS=$2
WW=/tmp/capstone/wt-wrap-supcall; MW=/tmp/capstone/wt-mon-supcall
FB=$HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot
XC=$FB/build-fpga/host/bin/riscv64-buildroot-linux-gnu-
CC=$HOME/dev/llvm-capstone/capstone/capstone-c/target/debug/capstone-c
rm -rf "$OUT"; mkdir -p "$OUT/opensbi" "$OUT/images"
git -C "$WW" archive HEAD | tar -x -C "$OUT/opensbi"
rm -rf "$OUT/opensbi/lib/sbi/capstone-sbi"; mkdir -p "$OUT/opensbi/lib/sbi/capstone-sbi"
git -C "$MW" archive HEAD | tar -x -C "$OUT/opensbi/lib/sbi/capstone-sbi"
cp "$FB/build-fpga/images/Image" "$FB/build-fpga/images/caplifive.dtb" "$OUT/images/"
for f in lib/sbi/sbi_capstone_dom.c lib/sbi/capstone_int_handler.c; do
  "$CC" --abi capstone "$OUT/opensbi/$f" -- -I"$OUT/opensbi/lib/sbi/capstone-sbi" -D__riscv_xlen=64 -DCAPSTONE_TARGET_FPGA $DEFS > "$OUT/opensbi/$f.S" 2> "$OUT/$(basename $f).err"
done
(cd "$OUT/opensbi" && taskset -c 0-7,32-39 nice -n 10 make -s PLATFORM=fpga/ariane CROSS_COMPILE="$XC" FW_PAYLOAD_PATH="$OUT/images/Image" \
   FW_FDT_PATH="$OUT/images/caplifive.dtb" FW_PAYLOAD_FDT_PATH="$OUT/images/caplifive.dtb" -j12 > "$OUT/make.log" 2>&1)
cp "$OUT/opensbi/build/platform/fpga/ariane/firmware/fw_payload.bin" "$OUT/fw_payload.bin"
printf 'fw %s  image %s  defs [%s]  wrapper %s  monitor %s\n' "$(sha256sum "$OUT/fw_payload.bin" | cut -c1-12)" \
  "$(sha256sum "$OUT/images/Image" | cut -c1-12)" "$DEFS" "$(git -C "$WW" rev-parse --short HEAD)" "$(git -C "$MW" rev-parse --short HEAD)"
