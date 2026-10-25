#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../tests/capstone-test-env.sh"
: "${KERNEL_BUILD:?Prepared kernel tree matching the booted Image}"
: "${CROSS_COMPILE:?RISC-V Linux compiler prefix}"
out=${1:?Output directory}
mkdir -p "$out/module"
cp "$here/module/Makefile" "$here/module/capstone_vm.c" "$here/wire.h" "$here/vm-abi.h" "$out/module/"
cp "$CAPSTONE_REPO_ROOT/capstone/capstone-qemu/target/riscv/cap_rev_table_abi.h" "$out/module/"
# The representability predicate, SHARED rather than copied: the module's arena test, grant() in
# the launcher and the allocator must agree on one rule, and that rule is cross-checked against
# the bounds encoder by tests/exact-bounds-iii/repr-cross-check.py. Copied from the tree being
# built, like the ABI header above, so a kit cannot pair a module and an SDK across revisions.
cp "$here/../include/capstone/representable.h" "$out/module/"
make -C "$KERNEL_BUILD" ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" \
    M="$(cd "$out/module" && pwd)" -j"${CAPSTONE_BUILD_JOBS:-12}" modules
"${CROSS_COMPILE}gcc" -O2 -static -pthread -Wall -Wextra -Werror \
    -I"$here/../include" "$here/exec.c" \
    "$here/../linux/"{application-image,image-hash,delegate-service,spawner,signals,park}.c \
    "$here/../linux/domain-fault.c" "$here/../linux/stub.S" "$here/../common/"{launch,delegate,spawn,msghdr}.c \
    -o "$out/capstone-vexec"

"${CROSS_COMPILE}gcc" -O2 -static -Wall -Wextra -Werror "$here/../linux/job.c" -o "$out/capstone-job"
