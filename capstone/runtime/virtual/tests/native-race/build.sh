#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?output directory}
: "${CROSS_COMPILE:?Linux cross compiler prefix}" "${CAPSTONE_NATIVE_CC:?copyable SDK capstone-cc}" "${KERNEL_BUILD:?prepared kernel}"
mkdir -p "$out"
CV_RACE_TEST=1 bash "$here/../../build-adapter.sh" "$out/adapter"
"${CROSS_COMPILE}gcc" -static -O2 "$here/pipe-fill.c" -o "$out/pipe-fill"
"$CAPSTONE_NATIVE_CC" -O2 -DCAPSTONE_RUNTIME_VIRTUAL -DCAPSTONE_NATIVE_SYSCALL_ABI \
 -I"$CAPSTONE_REPO_ROOT/capstone/runtime/include" "$here/race.c" "$here/rt-stubs.S" \
 "$CAPSTONE_REPO_ROOT/capstone/ports/musl-capstone/runtime/native-syscall.S" -o "$out/race.dom"
