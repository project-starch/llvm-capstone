#!/usr/bin/env bash
# The +Spatial twin of build-kernel-final.sh: the same kernel plus RISCV_CAPSTONE_SPATIAL_ONLY (../spatial-only-kernel/).
# The FINAL campaign Full kernel (Line B): Linux 6.1 + the series at 6d4504f8b073 + the runtime lane's msghdr loader
# (../msghdr-loader/sublet-msghdr-loader.patch), the board's fpgakernel.config + TASK/TRAP, no SMP/KSM/THP/modules,
# NO_HZ_IDLE at HZ=100 (as the Plain kernel), and RISCV_CAPSTONE_TRAP_TEST OFF: its syscall-number compare and its
# kernel page-fault hook are test-only code on hot paths that Plain lacks. Without TRAP_TEST the regs.S test loader is
# absent; SPEC and applications enter through the .dom loader. Tree: trapk/fpga-final, a copy of the board tree.
main() {
# CAPSTONE_HDD: the work disk's capstone directory; CAPSTONE_REPO_ROOT: the superproject checkout
: "${CAPSTONE_HDD:?}" "${CAPSTONE_REPO_ROOT:?}"
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W=${CAPSTONE_HDD}/trapk/fpga-final-spatial
H=${CAPSTONE_REPO_ROOT}/capstone/caplifive-system/sw/buildroot/build/host/bin
X=$H/riscv64-buildroot-linux-gnu-
cd "$W"
if [ ! -d linux-6.1/arch/riscv/kernel/sublet ]; then
  rm -rf linux-6.1
  tar -xf ${CAPSTONE_HDD}/ccsrrw-scan/src/linux-6.1.tar.xz
  cp ${CAPSTONE_REPO_ROOT}/capstone/caplifive-system/sw/buildroot/configs/fpgakernel.config linux-6.1/.config
  CAPSTONE_QEMU_SOURCE=${CAPSTONE_HDD}/ccsrrw-scan/qemu bash ../repo/capstone/kernel/apply.sh "$W/linux-6.1" > apply.log 2>&1
  (cd linux-6.1 && patch -p1 -s < "$here/../msghdr-loader/sublet-msghdr-loader.patch" && patch -p1 -s < "$here/../spatial-only-kernel/sublet-spatial-only.patch")
fi
cd linux-6.1
./scripts/config -e RISCV_CAPSTONE_TASK -e RISCV_CAPSTONE_TRAP -d RISCV_CAPSTONE_TRAP_TEST -e RISCV_CAPSTONE_SPATIAL_ONLY \
  -d SMP -d KSM -d TRANSPARENT_HUGEPAGE -d MODULES -d HZ_PERIODIC -e NO_HZ_IDLE -e NO_HZ_COMMON --set-str INITRAMFS_SOURCE "$W/initramfs.cpio"
make ARCH=riscv CROSS_COMPILE=$X olddefconfig > ../cfg.log 2>&1
grep -E '^CONFIG_(RISCV_CAPSTONE[A-Z_]*|SMP|KSM|TRANSPARENT_HUGEPAGE|MODULES|INITRAMFS_SOURCE|RISCV_ISA_C)=' .config > ../config-keys.txt || true
make ARCH=riscv CROSS_COMPILE=$X -j12 Image > ../build.log 2>&1
echo "BUILD-DONE rc=0" >> ../build.log
}
main "$@"; exit
