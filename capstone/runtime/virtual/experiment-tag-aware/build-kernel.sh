#!/bin/bash
# usage: build-kernel.sh SRC_TARBALL DIR CONFIG [extra config lines file]
set -euo pipefail
tb=$1 dir=$2 cfg=$3
L=$HOME/lanes/laneM
rm -rf "$dir"; mkdir -p "$dir"; cd "$dir"
zstd -dc "$tb" | tar xf -
cp "$cfg" .config
export LD_LIBRARY_PATH=$L/host/lib
export KBUILD_BUILD_USER=builder KBUILD_BUILD_HOST=capstone KBUILD_BUILD_VERSION=10 KBUILD_BUILD_TIMESTAMP="Tue Sep 29 11:27:43 +08 2026"
M="make ARCH=riscv CROSS_COMPILE=$L/host/bin/riscv64-buildroot-linux-gnu-"
if [ -n "${4:-}" ]; then ./scripts/kconfig/merge_config.sh -m .config "$4" >/dev/null 2>&1 || true; fi
$M olddefconfig >/dev/null
nice -n 5 $M -j24 Image modules_prepare > build.log 2>&1
$M -j24 modules >> build.log 2>&1 || true
sha256sum arch/riscv/boot/Image .config Module.symvers System.map
echo BUILD_DONE
