#!/usr/bin/env bash
# Build only on a designated build host. OUT and KERNEL_BUILD must be lane-owned.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
: "${OUT:?Set a private module output directory}"
: "${KERNEL_BUILD:?Set the matching prepared Linux build}"
: "${CROSS_COMPILE:?Set the RISC-V kernel toolchain prefix}"
mkdir -p "$OUT"
cp "$here/raw-page-probe.c" "$OUT/"
printf 'obj-m := raw-page-probe.o\n' > "$OUT/Makefile"
make -C "$KERNEL_BUILD" M="$(realpath "$OUT")" ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" modules
