#!/bin/bash
# Rebuild images/tickcost.ko against the board kernel (linux-6.4.14, build-fpga), the recipe cvprof.ko was built with:
# KCFLAGS="-march=rv64g -mabi=lp64d" so the module has no two-byte instructions (an RVC module hangs insmod on silicon).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
br=${BUILDROOT_FPGA:-$repo/capstone/caplifive-system/sw/buildroot/build-fpga}
out=${1:-$(mktemp -d)}
mkdir -p "$out"
cp "$here"/src/tickcost.c "$here"/src/Makefile "$out"/
make -C "$br/build/linux-6.4.14" M="$out" ARCH=riscv CROSS_COMPILE="$br/host/bin/riscv64-buildroot-linux-gnu-" \
     KCFLAGS="-march=rv64g -mabi=lp64d" modules
n2=$("$br/host/bin/riscv64-buildroot-linux-gnu-objdump" -d "$out/tickcost.ko" \
     | awk -F'\t' '/^ *[0-9a-f]+:\t/{w=$2; gsub(/ /,"",w); if(length(w)==4)c++} END{print c+0}')
echo "two-byte instructions: $n2"
[ "$n2" = 0 ] || { echo "REFUSED: compressed instructions in the module"; exit 1; }
sha256sum "$out/tickcost.ko"
