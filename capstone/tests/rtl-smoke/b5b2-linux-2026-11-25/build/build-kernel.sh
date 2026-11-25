#!/usr/bin/env bash
# The trap-route kernel for the FPGA (904cd9c6b): Linux 6.1 + the series at 6d4504f8b073, the board's
# fpgakernel.config + TASK/TRAP/TRAP_TEST, no SMP/KSM/THP/modules, initramfs from $W/initramfs.cpio.
main() {
# CAPSTONE_HDD: the work disk's capstone directory; CAPSTONE_REPO_ROOT: the superproject checkout
: "${CAPSTONE_HDD:?}" "${CAPSTONE_REPO_ROOT:?}"
set -euo pipefail
W=${CAPSTONE_HDD}/trapk/fpga
H=${CAPSTONE_REPO_ROOT}/capstone/caplifive-system/sw/buildroot/build/host/bin
X=$H/riscv64-buildroot-linux-gnu-
cd "$W"
if [ ! -d linux-6.1/arch/riscv/kernel/sublet ]; then
  rm -rf linux-6.1
  tar -xf ${CAPSTONE_HDD}/ccsrrw-scan/src/linux-6.1.tar.xz
  cp ${CAPSTONE_REPO_ROOT}/capstone/caplifive-system/sw/buildroot/configs/fpgakernel.config linux-6.1/.config
  CAPSTONE_QEMU_SOURCE=${CAPSTONE_HDD}/ccsrrw-scan/qemu bash ../repo/capstone/kernel/apply.sh "$W/linux-6.1" > apply.log 2>&1
fi
cd linux-6.1
./scripts/config -e RISCV_CAPSTONE_TASK -e RISCV_CAPSTONE_TRAP -e RISCV_CAPSTONE_TRAP_TEST \
  -d SMP -d KSM -d TRANSPARENT_HUGEPAGE -d MODULES --set-str INITRAMFS_SOURCE "$W/initramfs.cpio"
make ARCH=riscv CROSS_COMPILE=$X olddefconfig > ../cfg.log 2>&1
grep -E '^CONFIG_(RISCV_CAPSTONE[A-Z_]*|SMP|KSM|TRANSPARENT_HUGEPAGE|MODULES|INITRAMFS_SOURCE|RISCV_ISA_C)=' .config > ../config-keys.txt || true
make ARCH=riscv CROSS_COMPILE=$X -j12 Image > ../build.log 2>&1
echo "BUILD-DONE rc=0" >> ../build.log
}
main "$@"; exit
