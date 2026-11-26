#!/usr/bin/env bash
# make-payload.sh <overlay dir> <out dir>: a B5b-2 board image (fw_payload.bin) whose initramfs is the board's
# rootfs.cpio UNCHANGED (base-rootfs.cpio here) followed by <overlay dir> packed root-owned (cpio -R 0:0). The overlay
# must provide /init. Kernel = the trap-route FPGA kernel (build-kernel.sh, relinked with this initramfs), firmware =
# caplifive-opensbi trap-route-keep-tags (KEEP=n builds the stock control), DTB = caplifive-b5b2.dtb.
# DTB=<path> selects another device tree (caplifive-b5b2-wide.dtb adds sublet.dom_wide_grants=1, as the twin's
# trap_platform passes).
# Holds a lock: one kernel tree, one build at a time. Prints the image sha256 and the overlay's manifest.
# Portable copy of trapk/fpga-final/make-payload.sh: TREE=fpga-final (Full) or fpga-final-spatial (+Spatial), the trees
# build-kernel-final.sh and build-kernel-final-spatial.sh make. CAPSTONE_HDD: the work disk's capstone directory;
# CAPSTONE_REPO_ROOT: the superproject checkout. The campaign images take FW_EXTRA="FW_FAST_RDTIME=y".
main() {
set -euo pipefail
: "${CAPSTONE_HDD:?}" "${CAPSTONE_REPO_ROOT:?}"
W=${CAPSTONE_HDD}/trapk/${TREE:-fpga-final}
ovl=$(cd "${1:?overlay dir}" && pwd) out=${2:?out dir}
[ -x "$ovl/init" ] || { echo "no executable /init in $ovl" >&2; exit 1; }
mkdir -p "$out"; out=$(cd "$out" && pwd)
exec 9>"$W/.make-payload.lock"; flock 9
(cd "$ovl" && find . -mindepth 1 | sort | cpio -o -H newc -R 0:0 2>/dev/null) > "$out/overlay.cpio"
cat "$W/base-rootfs.cpio" "$out/overlay.cpio" > "$W/initramfs.cpio"
case ${TREE:-fpga-final} in fpga-final) k=build-kernel-final.sh ;; fpga-final-spatial) k=build-kernel-final-spatial.sh ;; *) echo "unknown TREE" >&2; exit 1 ;; esac
bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$k" > "$out/kernel.out" 2>&1
grep -q BUILD-DONE "$W/build.log" || { echo "kernel build failed: $W/build.log" >&2; exit 1; }
zcat "$W/linux-6.1/usr/initramfs_inc_data" | cmp - "$W/initramfs.cpio" || { echo "embedded initramfs differs" >&2; exit 1; }
cp "$W/initramfs.cpio" "$out/initramfs.cpio"
make -s -C ${CAPSTONE_HDD}/sbi-keep -j6 PLATFORM=fpga/ariane CROSS_COMPILE=riscv64-unknown-elf- \
  FW_CAPSTONE_KEEP_TAGS=${KEEP:-y} ${FW_EXTRA:-} FW_PAYLOAD_PATH="$W/linux-6.1/arch/riscv/boot/Image" FW_FDT_PATH="${DTB:-$W/caplifive-b5b2.dtb}" \
  O="$out/fw" > "$out/fw-build.log" 2>&1
cp "$out/fw/platform/fpga/ariane/firmware/fw_payload.bin" "$out/fw_payload.bin"
cp "$W/linux-6.1/arch/riscv/boot/Image" "$out/Image"
sz=$(stat -c %s "$out/fw_payload.bin"); [ "$sz" -le 26000000 ] || echo "WARNING: fw_payload.bin is $sz bytes, over the 26,000,000 kit cap" >&2
(cd "$out" && sha256sum fw_payload.bin Image overlay.cpio initramfs.cpio > SHA256SUMS)
(cd "$ovl" && find . -type f | sort | xargs sha256sum) > "$out/overlay-files.sha256"
fast=$(riscv64-unknown-elf-objdump -d "$out/fw/platform/fpga/ariane/firmware/fw_payload.elf" | grep -c fw_fast_mtime_addr || true)
# The campaign rule is FW_FAST_RDTIME=y in both arms (b5b2-linux RESULTS, boots 9-12); FW_EXTRA is the only way in.
[ "$fast" -gt 0 ] || echo "WARNING: no rdtime fast path in this firmware (FW_EXTRA=\"FW_FAST_RDTIME=y\" not given)" >&2
echo "fw_payload.bin $(cut -d' ' -f1 "$out/SHA256SUMS" | head -1) bytes $sz keep=${KEEP:-y} fast-path-refs=$fast"
}
main "$@"; exit
