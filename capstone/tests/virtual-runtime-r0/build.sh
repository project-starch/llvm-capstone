#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../capstone-test-env.sh"
: "${KERNEL_BUILD:?Prepared source/output matching the booted Image}"
: "${CROSS_COMPILE:?RISC-V Linux cross-compiler prefix}"
: "${CAPSTONE_CLANG:?Capstone clang binary}"
: "${CAPSTONE_LD_LLD:?Capstone linker}"
: "${CAPSTONE_OBJCOPY:?LLVM objcopy binary}"
out=${1:?Output directory}
mkdir -p "$out/module"
cp "$script_dir/"{Makefile,runtime_r0.c,wire.h} "$out/module/"
make -C "$KERNEL_BUILD" ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" \
    M="$(cd "$out/module" && pwd)" -j16 modules
"${CROSS_COMPILE}gcc" -O2 -static -Wall -Wextra "$script_dir/probe.c" \
    "$script_dir/program.S" -o "$out/probe"
"$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding -fno-builtin \
    -nostdinc -O2 -c "$script_dir/entry.c" -o "$out/c-entry.o"
"$CAPSTONE_LD_LLD" --image-base=0 -Ttext=0 -e r0_c_entry "$out/c-entry.o" -o "$out/c-entry.elf"
"$CAPSTONE_OBJCOPY" -O binary --only-section=.text "$out/c-entry.elf" "$out/c-entry.bin"
