#!/bin/bash
set -euo pipefail
W=${LANE_ROOT:-$HOME/lanes/progI}
export TMPDIR=$W/tmp CAPSTONE_TMP_ROOT=$W/tmp CAPSTONE_TOOLCHAIN_CHECKED=1
cd "$W/src"
source capstone/tests/capstone-test-env.sh
df -h "$W"
test "$(df -Pk "$W" | awk 'NR==2 {print $4}')" -ge 12582912
mkdir -p "$W/probes"
S=capstone/runtime/virtual/tests/native-integration
cc=$HOME/hosted-ab/xtc/bin/riscv64-buildroot-linux-gnu-gcc
"$cc" -static -O2 "$S/pipe-exec.c" -o "$W/probes/pipe-exec"
"$cc" -static -O2 "$S/budget.c" -o "$W/probes/budget-plain"
cc=$HOME/prof-dev/cap4/sdk/capstone-cc
"$cc" -O2 "$S/budget.c" -o "$W/probes/budget-legacy.dom"
"$cc" -O2 -DNATIVE_DIRECT -DCAPSTONE_RUNTIME_VIRTUAL -DCAPSTONE_NATIVE_SYSCALL_ABI -Icapstone/runtime/include \
 "$S/budget.c" capstone/ports/musl-capstone/runtime/native-syscall.S -o "$W/probes/budget-native.dom"
"$cc" -O2 -DCAPSTONE_RUNTIME_VIRTUAL -DCAPSTONE_NATIVE_SYSCALL_ABI -Icapstone/runtime/include \
 "$S/probe.c" capstone/ports/musl-capstone/runtime/native-syscall.S -o "$W/probes/probe.dom"
sha256sum "$W/probes/pipe-exec" "$W/probes/probe.dom" "$W"/probes/budget-* > "$W/probes/inputs.sha256"
