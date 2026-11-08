#!/usr/bin/env bash
# MEASUREMENT ONLY. build-firmware.sh OPENSBI_GIT OUT: QEMU's bundled OpenSBI (capstone-qemu roms/opensbi,
# 057eb10b6d52, the native platform's firmware) plus opensbi-057eb10b-cscheck-emu.patch, which emulates
# CSCHECKR/W on the illegal-instruction trap from S (custom-2 0x5b, funct3 1, funct7 0x25/0x26) and adds an
# experimental SBI extension 0x08435343 (fid 0 probe, 1 read counter a0, 2 reset) that the adapter's
# native_counters parameter reads. Boot it as the -bios of a model run with x-capstone-checked-copy=false and
# the kernel argument capstone_vm.checked_copy_emu=1. The emulation's shortcuts (the mutable +64+16r slot,
# the frame-word epoch at +2336 and classification at +2344) are listed in its source; see
# docs/history/08-11-2026_14-30-00_native-checked-emu-measurement.md.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
src=${1:?OpenSBI git repository holding 057eb10b6d52 (capstone-qemu roms/opensbi)}
out=${2:?new output directory}
: "${CROSS_COMPILE:?riscv64 Linux or ELF compiler prefix}"
mkdir "$out"
git -C "$src" archive 057eb10b6d523540012e6947d5c9f63e95244e94 | tar -x -C "$out"
patch -d "$out" -p1 < "$here/opensbi-057eb10b-cscheck-emu.patch"
make -C "$out" -j"${JOBS:-12}" CROSS_COMPILE="$CROSS_COMPILE" PLATFORM=generic O=build
sha256sum "$out/build/platform/generic/firmware/fw_dynamic.bin"
