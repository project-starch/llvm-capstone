#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../tests/capstone-test-env.sh"
: "${KERNEL_BUILD:?Prepared kernel tree matching the booted Image}"
: "${CROSS_COMPILE:?RISC-V Linux compiler prefix}"
out=${1:?Output directory}
bash "$here/stage-adapter.sh" "$out/module"
make -C "$KERNEL_BUILD" ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" \
    M="$(cd "$out/module" && pwd)" -j"${CAPSTONE_BUILD_JOBS:-12}" modules
torture_flags=()
[[ ${CAPSTONE_VM_TORTURE:-0} != 1 ]] || torture_flags+=(-DCAPSTONE_VM_TORTURE)
"${CROSS_COMPILE}gcc" "${torture_flags[@]}" -O2 -static -pthread -Wall -Wextra -Werror \
    -I"$here/../include" "$here/exec.c" \
    "$here/../linux/"{application-image,image-hash,delegate-service,spawner,signals,park}.c \
    "$here/../linux/domain-fault.c" "$here/../linux/stub.S" "$here/../common/"{launch,delegate,spawn,msghdr}.c \
    -o "$out/capstone-vexec"

"${CROSS_COMPILE}gcc" -O2 -static -Wall -Wextra -Werror "$here/../linux/job.c" -o "$out/capstone-job"
