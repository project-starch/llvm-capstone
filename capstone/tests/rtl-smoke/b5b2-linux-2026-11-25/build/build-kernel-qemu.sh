#!/usr/bin/env bash
# The trap-route kernel for the firmware bring-up: Linux 6.1 + the collaborator's series at
# origin/native-kernel-new-qemu 6d4504f8b073, the scan kernel's config + TRAP and TRAP_TEST.
main() {
# CAPSTONE_HDD: the work disk's capstone directory; CAPSTONE_REPO_ROOT: the superproject checkout
: "${CAPSTONE_HDD:?}" "${CAPSTONE_REPO_ROOT:?}"
set -euo pipefail
W=${CAPSTONE_HDD}/trapk
cd "$W"
rm -rf linux-6.1 repo
tar -xf ${CAPSTONE_HDD}/ccsrrw-scan/src/linux-6.1.tar.xz
mkdir repo
git -C ${CAPSTONE_REPO_ROOT} archive 6d4504f8b073 capstone/kernel capstone/runtime | tar -x -C repo
cp ${CAPSTONE_HDD}/ccsrrw-scan/src/linux-6.1/.config linux-6.1/.config
CAPSTONE_QEMU_SOURCE=${CAPSTONE_HDD}/ccsrrw-scan/qemu bash repo/capstone/kernel/apply.sh "$W/linux-6.1" > apply.log 2>&1
cd linux-6.1
./scripts/config -e RISCV_CAPSTONE_TASK -e RISCV_CAPSTONE_TRAP -e RISCV_CAPSTONE_TRAP_TEST -d SMP -d KSM -d TRANSPARENT_HUGEPAGE -d MODULES
make ARCH=riscv CROSS_COMPILE=${CAPSTONE_REPO_ROOT}/capstone/caplifive-system/sw/buildroot/build/host/bin/riscv64-buildroot-linux-gnu- olddefconfig > ../cfg.log 2>&1
grep -E '^CONFIG_(RISCV_CAPSTONE[A-Z_]*|SMP|KSM|TRANSPARENT_HUGEPAGE|MODULES)=' .config > ../config-keys.txt || true
make ARCH=riscv CROSS_COMPILE=${CAPSTONE_REPO_ROOT}/capstone/caplifive-system/sw/buildroot/build/host/bin/riscv64-buildroot-linux-gnu- -j12 Image > ../build.log 2>&1
echo "BUILD-DONE rc=0" >> ../build.log
}
main "$@"; exit
