#!/bin/bash
# Run on p12 after exporting this directory into ~/lanes/laneAG2/contracts.
set -euo pipefail
W=${LANE_ROOT:-$HOME/lanes/laneAG2}
cd "$W/src"
export CAPSTONE_TOOLCHAIN_CHECKED=1 CAPSTONE_TMP_ROOT=$W/tmp CAPSTONE_QEMU_LOCK=$W/build.lock
source capstone/tests/capstone-test-env.sh
df -h "$W"
"${CAPSTONE_CC:-$HOME/prof-dev/cap4/sdk/capstone-cc}" -O2 -I"$W/src/capstone/runtime/include" "$W/contracts/contract.c" -Wl,--wrap=__capstone_vm_delegate_transport -o "$W/contracts/contract.dom"
"${NATIVE_CC:-$HOME/prof-dev/rv/rv-cc}" -O2 "$W/contracts/driver.c" -o "$W/contracts/driver"
"${NATIVE_CC:-$HOME/prof-dev/rv/rv-cc}" -O2 "$W/contracts/slot-wait.c" -o "$W/contracts/slot-wait"
"${NATIVE_CC:-$HOME/prof-dev/rv/rv-cc}" -O2 "$W/contracts/slot-progress.c" -o "$W/contracts/slot-progress"
"${CAPSTONE_CC:-$HOME/prof-dev/cap4/sdk/capstone-cc}" -O2 "$W/contracts/scalar-memory.c" -o "$W/contracts/scalar-memory.dom"
"${CAPSTONE_CC:-$HOME/prof-dev/cap4/sdk/capstone-cc}" -O2 "$W/contracts/wide-grants.c" -o "$W/contracts/wide-grants.dom"
"${CAPSTONE_CC:-$HOME/prof-dev/cap4/sdk/capstone-cc}" -O2 -I"$W/src/capstone/runtime/include" \
 "$W/contracts/resource-probe.c" -Wl,--wrap=__capstone_vm_delegate_transport -o "$W/contracts/resource-probe.dom"
# This helper needs Linux UAPI capability/securebits headers as well as libc.
"${DAC_CC:-$HOME/hosted-ab/xtc/bin/riscv64-buildroot-linux-gnu-gcc}" -static -O2 \
 "$W/contracts/dac-driver.c" -o "$W/contracts/dac-driver"
python3 "$W/contracts/malformed.py" "$W/contracts/contract.dom"
