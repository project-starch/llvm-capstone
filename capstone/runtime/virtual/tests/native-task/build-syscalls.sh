#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?output directory}
: "${CROSS_COMPILE:?Linux compiler prefix}" "${CAPSTONE_NATIVE_CC:?copyable SDK compiler}"
mkdir -p "$out"
source_dir="$here/../native-integration"
"${CROSS_COMPILE}gcc" -O2 -static "$source_dir/budget.c" -o "$out/budget-plain"
"${CROSS_COMPILE}gcc" -O2 -static "$source_dir/pipe-exec.c" -o "$out/pipe-exec"
"$CAPSTONE_NATIVE_CC" -O2 "$source_dir/budget.c" -o "$out/budget-exchange.dom"
"$CAPSTONE_NATIVE_CC" -O2 -DFORCE_EXCHANGE_PID "$source_dir/budget.c" -o "$out/budget-exchange-scalar.dom"
"$CAPSTONE_NATIVE_CC" -O2 -DNATIVE_DIRECT -DCAPSTONE_RUNTIME_VIRTUAL -DCAPSTONE_NATIVE_SYSCALL_ABI \
    -I"$CAPSTONE_REPO_ROOT/capstone/runtime/include" "$source_dir/budget.c" \
    "$CAPSTONE_REPO_ROOT/capstone/ports/musl-capstone/runtime/native-syscall.S" -o "$out/budget-register.dom"
