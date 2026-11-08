#!/usr/bin/env bash
# Source capstone/tests/capstone-test-env.sh before invoking this script.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 || ( $# == 4 && $4 != --lto ) ]]; then
    echo "Usage: $0 QEMU_GIT_REPOSITORY REVISION FRESH_OUTPUT_DIRECTORY [--lto]" >&2
    exit 2
fi
qemu_repo=$(realpath "$1")
qemu_rev=$(git -C "$qemu_repo" rev-parse --verify "$2^{commit}")
qemu_out=$(realpath -m "$3")
mkdir "$qemu_out"
mkdir "$qemu_out/src" "$qemu_out/build"
printf '%s\n' "$qemu_rev" > "$qemu_out/revision.txt"
git -C "$qemu_repo" archive "$qemu_rev" | tar -x -C "$qemu_out/src"
options=(--target-list=riscv64-softmmu --disable-docs --disable-werror
         --disable-tools --disable-guest-agent --disable-slirp --disable-plugins)
if [[ ${4:-} == --lto ]]; then
    options+=(--enable-lto)
fi
(
    cd "$qemu_out/build"
    printf '%q ' ../src/configure "${options[@]}" > configure-command.txt
    printf '\n' >> configure-command.txt
    ../src/configure "${options[@]}" > configure.log 2>&1
    if [[ ${4:-} == --lto ]]; then
        ./pyvenv/bin/meson configure "-Db_lto_threads=${QEMU_LTO_THREADS:-8}" > meson-lto.log 2>&1
    fi
    ninja -j"${QEMU_BUILD_JOBS:-10}" qemu-system-riscv64 > build.log 2>&1
    sha256sum qemu-system-riscv64 > binary.sha256
)
