#!/bin/bash
# Build the guest programs and the race-test module/adapter. LANE_ROOT holds src/,
# kernel/ (the booted kernel's build tree), platform/ and tmp/ (see run.sh).
set -euo pipefail
W=${LANE_ROOT:-$HOME/lanes/progRT}
export TMPDIR=$W/tmp CAPSTONE_TMP_ROOT=$W/tmp CAPSTONE_TOOLCHAIN_CHECKED=1 CAPSTONE_BUILD_JOBS=8
cd "$W/src"
source capstone/tests/capstone-test-env.sh
test "$(df -Pk "$W" | awk 'NR==2 {print $4}')" -ge 12582912
export KERNEL_BUILD=$W/kernel CROSS_COMPILE=$HOME/hosted-ab/xtc/bin/riscv64-buildroot-linux-gnu-
CV_RACE_TEST=1 bash capstone/runtime/virtual/build-adapter.sh "$W/adapter-race"
S=capstone/runtime/virtual/tests/native-race
mkdir -p "$W/probes"
"${CROSS_COMPILE}gcc" -static -O2 "$S/pipe-fill.c" -o "$W/probes/pipe-fill"
"$HOME/prof-dev/cap4/sdk/capstone-cc" -O2 -DCAPSTONE_RUNTIME_VIRTUAL -DCAPSTONE_NATIVE_SYSCALL_ABI \
  -Icapstone/runtime/include "$S/race.c" "$S/rt-stubs.S" \
  capstone/ports/musl-capstone/runtime/native-syscall.S -o "$W/probes/race.dom"
