#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../tests/capstone-test-env.sh"
: "${KERNEL_BUILD:?Prepared kernel tree matching the booted Image}"
: "${CROSS_COMPILE:?RISC-V Linux compiler prefix}"
out=${1:?Output directory}
mkdir -p "$out/module"
cp "$here/module/Makefile" "$here/module/capstone_vm.c" "$here/wire.h" "$here/vm-abi.h" "$out/module/"
cp "$here/heap-native.inc" "$out/module/"
cp "$CAPSTONE_REPO_ROOT/capstone/capstone-qemu/target/riscv/cap_rev_table_abi.h" "$out/module/"
make -C "$KERNEL_BUILD" ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" \
    M="$(cd "$out/module" && pwd)" -j16 modules
native=$(bash "$here/build-native-musl.sh" "$out/native-musl" | tail -1)
native_uapi=$("${CROSS_COMPILE}gcc" -print-sysroot)/usr/include
native_atomic=$(dirname "$("${CROSS_COMPILE}gcc" -print-file-name=libatomic.a)")
"${CROSS_COMPILE}gcc" -O2 -static -pthread -Wall -Wextra -Werror \
    -specs="$native/lib/musl-gcc.specs" \
    -idirafter "$native_uapi" \
    -L"$native_atomic" \
    -Wl,--wrap=memcpy,--wrap=munmap,--wrap=__munmap,--wrap=__mremap,--wrap=__libc_free \
    -I"$here/../include" "$here/exec.c" \
    "$here/native-heap.c" \
    "$here/../linux/"{application-image,image-hash,delegate-service,spawner,signals,park}.c \
    "$here/../linux/domain-fault.c" "$here/../linux/stub.S" "$here/../common/"{launch,delegate,spawn,msghdr}.c \
    -o "$out/capstone-vexec"

"${CROSS_COMPILE}gcc" -O2 -static -Wall -Wextra -Werror "$here/../linux/job.c" -o "$out/capstone-job"
