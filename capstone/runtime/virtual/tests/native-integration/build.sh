#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?output directory}
: "${CROSS_COMPILE:?Linux cross compiler prefix}" "${CAPSTONE_NATIVE_CC:?copyable SDK capstone-cc}"
mkdir -p "$out"
"${CROSS_COMPILE}gcc" -static -O2 "$here/pipe-exec.c" -o "$out/pipe-exec"
"${CROSS_COMPILE}gcc" -static -O2 "$here/budget.c" -o "$out/budget-plain"
"$CAPSTONE_NATIVE_CC" -O2 "$here/budget.c" -o "$out/budget-legacy.dom"
flags=(-O2 -DCAPSTONE_RUNTIME_VIRTUAL -DCAPSTONE_NATIVE_SYSCALL_ABI -I"$CAPSTONE_REPO_ROOT/capstone/runtime/include")
stub="$CAPSTONE_REPO_ROOT/capstone/ports/musl-capstone/runtime/native-syscall.S"
"$CAPSTONE_NATIVE_CC" "${flags[@]}" -DNATIVE_DIRECT "$here/budget.c" "$stub" -o "$out/budget-native.dom"
"$CAPSTONE_NATIVE_CC" "${flags[@]}" "$here/probe.c" "$stub" -o "$out/probe.dom"
"${CROSS_COMPILE}gcc" -static -O2 -Wall -Wextra -Werror "$here/entry-driver.c" -o "$out/entry-driver"
"$CAPSTONE_NATIVE_CC" "${flags[@]}" "$here/blocked.c" "$stub" -o "$out/blocked.dom"
"$CAPSTONE_NATIVE_CC" -O2 "$here/gcc-coverage.c" -o "$out/gcc-coverage.dom"
sha256sum "$out/gcc-coverage.dom" "$out/pipe-exec" "$out/probe.dom" "$out/blocked.dom" "$out/entry-driver" "$out"/budget-* > "$out/inputs.sha256"
