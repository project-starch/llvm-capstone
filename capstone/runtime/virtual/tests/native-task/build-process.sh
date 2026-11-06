#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?output directory}
: "${CAPSTONE_EXCHANGE_CC:?exchange SDK compiler}" "${CAPSTONE_NATIVE_CC:?direct IO SDK compiler}" "${CROSS_COMPILE:?Linux compiler prefix}"
mkdir -p "$out"
"$CAPSTONE_EXCHANGE_CC" -O2 -I"$CAPSTONE_REPO_ROOT/capstone/runtime/include" "$here/process-services.c" -o "$out/process-exchange.dom"
"$CAPSTONE_NATIVE_CC" -O2 -DPROCESS_DIRECT -I"$CAPSTONE_REPO_ROOT/capstone/runtime/include" "$here/process-services.c" -o "$out/process-direct.dom"
"${CROSS_COMPILE}gcc" -static -O2 -Wall -Wextra -Werror "$here/process-driver.c" -o "$out/process-driver"
sha256sum "$here/process-services.c" "$here/process-driver.c" "$out"/*.dom "$out/process-driver" > "$out/process-inputs.sha256"
