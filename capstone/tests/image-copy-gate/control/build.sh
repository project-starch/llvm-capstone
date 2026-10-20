#!/usr/bin/env bash
# Links control/byte-copy.c into a freestanding capstone64 image: build.sh <llvm-bin> <out.elf>.
set -euo pipefail
B=${1:?llvm-bin}; O=${2:?out.elf}; D=$(cd "$(dirname "$0")" && pwd)
"$B/clang" -target capstone64-unknown-elf -ffreestanding -nostdlibinc -O2 -c "$D/byte-copy.c" -o "$O.o"
"$B/ld.lld" -static -e _start "$O.o" -o "$O"
