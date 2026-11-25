#!/usr/bin/env bash
# run-trap.sh <fw_jump.elf> <name> [sstc on|off]: the collaborator's trap-route test (regs.S + its /init, boot.py
# unchanged) on the trap-route kernel. The initramfs is the board's buildroot rootfs.cpio UNCHANGED followed by a
# root-owned overlay (cpio -R 0:0): extracting and repacking it as a user made the setuid busybox run with euid 1000
# (2026-11-25), and the test loader then refused with EPERM. INIT and EXTRA (space-separated) override /init and add files.
main() {
: "${CAPSTONE_HDD:?}" "${CAPSTONE_REPO_ROOT:?}"
set -euo pipefail
W=${CAPSTONE_HDD}/trapk
fw=${1:?fw} name=${2:?name} sstc=${3:-off}
T=$W/repo/capstone/kernel/tests/trap-route
work=$W/run-$name; rm -rf "$work"; mkdir -p "$work/ovl"
riscv64-unknown-elf-gcc -nostdlib -static -march=rv64gc -mabi=lp64d -o "$work/ovl/regs" "$T/regs.S"
install -m 0755 "${INIT:-$T/init}" "$work/ovl/init"
for f in ${EXTRA:-}; do install -m 0755 "$f" "$work/ovl/"; done
(cd "$work/ovl" && ls | cpio -o -H newc -R 0:0 2>/dev/null) > "$work/overlay.cpio"
cat $W/fpga/base-rootfs.cpio "$work/overlay.cpio" | gzip -9 > "$work/initramfs.cpio.gz"
python3 "$T/boot.py" ${CAPSTONE_HDD}/qemu-crv/build/qemu-system-riscv64 "$fw" \
    "$W/linux-6.1/arch/riscv/boot/Image" "$work/initramfs.cpio.gz" "$work/boot.log" "$sstc" > "$work/verdict.txt"
cat "$work/verdict.txt"
}
main "$@"; exit
