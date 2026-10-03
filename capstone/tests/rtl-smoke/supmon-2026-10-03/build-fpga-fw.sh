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
(cd "$OUT/opensbi" && taskset -c 0-7,32-39 nice -n 10 make -s PLATFORM=fpga/ariane CROSS_COMPILE="$XC" CAPSTONE_PLATFORM_DEFS="$DEFS" FW_PAYLOAD_PATH="$OUT/images/Image" \
   FW_FDT_PATH="$OUT/images/caplifive.dtb" FW_PAYLOAD_FDT_PATH="$OUT/images/caplifive.dtb" -j12 > "$OUT/make.log" 2>&1)
cp "$OUT/opensbi/build/platform/fpga/ariane/firmware/fw_payload.bin" "$OUT/fw_payload.bin"
# GATE (2026-10-03, board boot supmon-c5): dom_init carves every monitor global from dom_stack before any trap
# vector exists, so globals + 2 KiB of real stack must fit, or the board hangs silently after OpenSBI's banner.
# (2 KiB: the board build boots with 3,232 B left -- boot supmon-c3 -- so the margin is set below what is proven.)
# And the firmware RW (to _fw_end) + heap/scratch (36 KiB, from the boot banner) must stay inside the 128 KiB
# M-mode region the DTS reserves (0x80080000-0x8009ffff).
python3 - "$OUT" "${XC}nm" <<'PY'
import sys, re, subprocess
out, nm = sys.argv[1], sys.argv[2]
s = open(out + "/opensbi/lib/sbi/sbi_capstone_dom.c.S").read()
m = re.search(r"^dom_init:\n(.*?)^[A-Za-z_][A-Za-z0-9_]*:$", s, re.S | re.M)
carve = sum(int(x) for x in re.findall(r"addi t1, t1, -(\d+)", m.group(1)))
sym = {}
for l in subprocess.run([nm, out + "/opensbi/build/platform/fpga/ariane/firmware/fw_payload.elf"],
                        capture_output=True, text=True).stdout.splitlines():
    f = l.split()
    if len(f) == 3: sym[f[2]] = int(f[0], 16)
stack = sym["dom_stack_end"] - sym["dom_stack"]
fw_end = sym["_fw_end"]
print(f"gate: dom_init carves {carve} B of a {stack} B dom_stack; _fw_end {fw_end:#x}")
if carve + 2048 > stack: sys.exit(f"GATE FAIL: globals {carve} B + 2048 B stack > dom_stack {stack} B")
if fw_end + 0x9000 > 0x800A0000: sys.exit(f"GATE FAIL: _fw_end {fw_end:#x} + 36 KiB heap/scratch crosses 0x800A0000")
print("gate: PASS")
PY
printf 'fw %s  image %s  defs [%s]  wrapper %s  monitor %s\n' "$(sha256sum "$OUT/fw_payload.bin" | cut -c1-12)" \
  "$(sha256sum "$OUT/images/Image" | cut -c1-12)" "$DEFS" "$(git -C "$WW" rev-parse --short HEAD)" "$(git -C "$MW" rev-parse --short HEAD)"
