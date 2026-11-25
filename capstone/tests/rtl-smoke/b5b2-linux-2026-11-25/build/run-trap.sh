#!/usr/bin/env bash
# run-trap.sh <fw_jump.elf> <name> [sstc on|off]: the collaborator's trap-route test (regs.S + its /init,
# boot.py unchanged) on the trap-route kernel, with the board buildroot rootfs as the busybox base.
main() {
# CAPSTONE_HDD: the work disk's capstone directory; CAPSTONE_REPO_ROOT: the superproject checkout
: "${CAPSTONE_HDD:?}" "${CAPSTONE_REPO_ROOT:?}"
set -euo pipefail
W=${CAPSTONE_HDD}/trapk
fw=${1:?fw} name=${2:?name} sstc=${3:-off}
T=$W/repo/capstone/kernel/tests/trap-route
work=$W/run-$name; rm -rf "$work"; mkdir -p "$work/root"
riscv64-unknown-elf-gcc -nostdlib -static -march=rv64gc -mabi=lp64d -o "$work/regs" "$T/regs.S"
(cd "$work/root" && cpio -id < ${CAPSTONE_REPO_ROOT}/capstone/caplifive-system/sw/buildroot/build/images/rootfs.cpio 2>/dev/null || true)
install -m 0755 "$work/regs" "$work/root/regs"
install -m 0755 "${INIT:-$T/init}" "$work/root/init"
[ -n "${EXTRA:-}" ] && install -m 0755 "$EXTRA" "$work/root/"
(cd "$work/root" && find . | cpio -o -H newc 2>/dev/null | gzip -9) > "$work/initramfs.cpio.gz"
python3 "$T/boot.py" ${CAPSTONE_HDD}/qemu-crv/build/qemu-system-riscv64 "$fw" \
    "$W/linux-6.1/arch/riscv/boot/Image" "$work/initramfs.cpio.gz" "$work/boot.log" "$sstc" > "$work/verdict.txt"
cat "$work/verdict.txt"
}
main "$@"; exit
