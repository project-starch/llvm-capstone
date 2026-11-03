#!/bin/bash
# Run on p12 after exporting this directory into ~/lanes/laneAG/contracts.
set -euo pipefail
W=~/lanes/laneAG
cd "$W/src"
export CAPSTONE_TOOLCHAIN_CHECKED=1 CAPSTONE_TMP_ROOT=$W/tmp CAPSTONE_QEMU_LOCK=$W/build.lock
source capstone/tests/capstone-test-env.sh
df -h "$W"
"${CAPSTONE_CC:-$HOME/prof-dev/cap4/sdk/capstone-cc}" -O2 "$W/contracts/contract.c" -o "$W/contracts/contract.dom"
"${NATIVE_CC:-$HOME/prof-dev/rv/rv-cc}" -O2 "$W/contracts/driver.c" -o "$W/contracts/driver"
python3 "$W/contracts/malformed.py" "$W/contracts/contract.dom"
