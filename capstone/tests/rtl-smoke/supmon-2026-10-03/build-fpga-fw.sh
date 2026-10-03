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
# FW_IMAGE_DIR: build on a saved Image + dtb instead of the shared build's current ones (the shared image can be
# restored for other lanes while private firmwares keep using a staged Image).
IMGDIR=${FW_IMAGE_DIR:-$FB/build-fpga/images}
cp "$IMGDIR/Image" "$IMGDIR/caplifive.dtb" "$OUT/images/"
for f in lib/sbi/sbi_capstone_dom.c lib/sbi/capstone_int_handler.c; do
  "$CC" --abi capstone "$OUT/opensbi/$f" -- -I"$OUT/opensbi/lib/sbi/capstone-sbi" -D__riscv_xlen=64 -DCAPSTONE_TARGET_FPGA $DEFS > "$OUT/opensbi/$f.S" 2> "$OUT/$(basename $f).err"
done
# FW_POSTCALL ("fence" or "nop8"): insert that sequence after EVERY domcall in the generated monitor -- the workaround
# under test for the post-CALL `ccsrrw sp <- cscratch; ldc` hang (sup-resume-2026-10-03). Gated on the ARTIFACT after
# the build: every CALL word in the firmware ELF must be followed by the inserted sequence.
# FW_PRECALL=fence: a fence immediately BEFORE every domcall -- the S-16 workaround (fpga-repros/S16-...): the switch
# must never start while the store buffer's commit queue is full. Gated on the artifact below.
if [ -n "${FW_PRECALL:-}" ]; then
  python3 - "$OUT/opensbi/lib/sbi/sbi_capstone_dom.c.S" <<'PY'
import sys, re
p = sys.argv[1]
out, n = [], 0
for line in open(p).read().split("\n"):
    if re.match(r"^\s*domcall\(", line):
        out.append("  fence"); n += 1
    out.append(line)
if n == 0: sys.exit("PRECALL GATE FAIL: no domcall found in the generated monitor")
open(p, "w").write("\n".join(out)); print(f"precall: fence before {n} domcall sites")
PY
fi
if [ -n "${FW_POSTCALL:-}" ]; then
  python3 - "$OUT/opensbi/lib/sbi/sbi_capstone_dom.c.S" "$FW_POSTCALL" <<'PY'
import sys, re
p, kind = sys.argv[1], sys.argv[2]
seq = {"fence": ["  fence"], "nop8": ["  nop"] * 8}[kind]
out, n = [], 0
for line in open(p).read().split("\n"):
    out.append(line)
    if re.match(r"^\s*domcall\(", line):
        out.extend(seq); n += 1
if n == 0: sys.exit("POSTCALL GATE FAIL: no domcall found in the generated monitor")
open(p, "w").write("\n".join(out)); print(f"postcall: {kind} after {n} domcall sites")
PY
fi
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
if [ -n "${FW_PRECALL:-}" ]; then
  python3 - "$OUT/opensbi/build/platform/fpga/ariane/firmware/fw_payload.elf" "${XC}objcopy" <<'PY'
import sys, subprocess, struct, tempfile, os
elf, objcopy = sys.argv[1:3]
tmp = tempfile.mktemp()
subprocess.run([objcopy, "-O", "binary", "--only-section=.text", elf, tmp], check=True)
b = open(tmp, "rb").read(); os.remove(tmp)
calls = bad = 0
for off in range(4, len(b) - 4, 2):
    w = struct.unpack_from("<I", b, off)[0]
    if (w & 0x7f) == 0x5b and ((w >> 12) & 7) == 1 and (w >> 25) == 0x20:
        calls += 1; bad += (struct.unpack_from("<I", b, off - 4)[0] != 0x0ff0000f)
print(f"precall gate: {calls} CALL words, {bad} not preceded by a fence")
if calls == 0 or bad: sys.exit("PRECALL GATE FAIL")
PY
fi
if [ -n "${FW_POSTCALL:-}" ]; then
  python3 - "$OUT/opensbi/build/platform/fpga/ariane/firmware/fw_payload.elf" "$FW_POSTCALL" "${XC}objcopy" <<'PY'
import sys, subprocess, struct, tempfile, os
elf, kind, objcopy = sys.argv[1:4]
want = {"fence": [0x0ff0000f], "nop8": [0x00000013] * 8}[kind]
tmp = tempfile.mktemp()
subprocess.run([objcopy, "-O", "binary", "--only-section=.text", elf, tmp], check=True)
b = open(tmp, "rb").read(); os.remove(tmp)
calls = bad = 0
for off in range(0, len(b) - 4, 2):            # 2-byte steps: the firmware mixes compressed code
    w = struct.unpack_from("<I", b, off)[0]
    if (w & 0x7f) == 0x5b and ((w >> 12) & 7) == 1 and (w >> 25) == 0x20:
        nxt = [struct.unpack_from("<I", b, off + 4 + 4 * i)[0] for i in range(len(want))]
        calls += 1; bad += (nxt != want)
print(f"postcall gate: {calls} CALL words, {bad} without the {kind} sequence")
if calls == 0 or bad: sys.exit("POSTCALL GATE FAIL")
PY
fi
printf 'fw %s  image %s  defs [%s]  wrapper %s  monitor %s\n' "$(sha256sum "$OUT/fw_payload.bin" | cut -c1-12)" \
  "$(sha256sum "$OUT/images/Image" | cut -c1-12)" "$DEFS" "$(git -C "$WW" rev-parse --short HEAD)" "$(git -C "$MW" rev-parse --short HEAD)"
