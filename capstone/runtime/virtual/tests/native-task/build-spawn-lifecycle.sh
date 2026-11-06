#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?output directory}
: "${CAPSTONE_EXCHANGE_CC:?exchange SDK compiler}" "${CAPSTONE_NATIVE_CC:?direct IO SDK compiler}" "${CROSS_COMPILE:?Linux compiler prefix}"
mkdir -p "$out"
"$CAPSTONE_EXCHANGE_CC" -O2 -I"$CAPSTONE_REPO_ROOT/capstone/runtime/include" "$here/spawn-lifecycle.c" -o "$out/spawn-lifecycle-exchange.dom"
"$CAPSTONE_NATIVE_CC" -O2 -DSPAWN_DIRECT -I"$CAPSTONE_REPO_ROOT/capstone/runtime/include" "$here/spawn-lifecycle.c" -o "$out/spawn-lifecycle-direct.dom"
"${CROSS_COMPILE}gcc" -static -O2 -Wall -Wextra -Werror "$here/spawn-driver.c" -o "$out/spawn-driver"
sha256sum "$here/spawn-lifecycle.c" "$here/spawn-driver.c" "$here/build-spawn-lifecycle.sh" \
    "$out/spawn-lifecycle-exchange.dom" "$out/spawn-lifecycle-direct.dom" "$out/spawn-driver" > "$out/spawn-lifecycle-inputs.sha256"
