#!/usr/bin/env bash
# Source capstone/tests/capstone-test-env.sh before invoking this script.
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "Usage: $0 QEMU_GIT_REPOSITORY FRESH_OUTPUT_DIRECTORY" >&2
    exit 2
fi
qemu_repo=$(realpath "$1")
qemu_out=$(realpath -m "$2")
mkdir "$qemu_out"
stock_rev=6bb4a8a47a43f35a345f107227fcd6abed59e62c
fork_rev=dd041bac5ad9c145c3872a50804ca714a26b1c88
for arm in stock fork; do
    rev=$stock_rev
    [[ $arm != fork ]] || rev=$fork_rev
    mkdir "$qemu_out/$arm-src" "$qemu_out/$arm-build"
    git -C "$qemu_repo" archive "$rev" | tar -x -C "$qemu_out/$arm-src"
    printf '%s\n' "$rev" > "$qemu_out/$arm-revision.txt"
    (
        cd "$qemu_out/$arm-build"
        ../"$arm-src"/configure --target-list=riscv64-softmmu --disable-docs \
            --disable-werror --disable-tools --disable-guest-agent \
            --disable-slirp --disable-plugins > configure.log 2>&1
        ninja -j"${QEMU_BUILD_JOBS:-12}" qemu-system-riscv64 > build.log 2>&1
        sha256sum qemu-system-riscv64 > binary.sha256
    )
done
