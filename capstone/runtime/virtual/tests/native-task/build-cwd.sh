#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?output directory}
: "${CAPSTONE_EXCHANGE_CC:?exchange SDK compiler}" "${CAPSTONE_NATIVE_CC:?direct IO SDK compiler}" "${CROSS_COMPILE:?Linux compiler prefix}"
mkdir -p "$out"
"$CAPSTONE_EXCHANGE_CC" -O2 "$here/cwd.c" -o "$out/cwd-exchange.dom"
"$CAPSTONE_NATIVE_CC" -O2 -DCWD_RAW "$here/cwd.c" -o "$out/cwd-direct.dom"
"${CROSS_COMPILE}gcc" -static -O2 -Wall -Wextra -Werror "$here/cwd-driver.c" -o "$out/cwd-driver"
sha256sum "$here/cwd.c" "$here/cwd-driver.c" "$out"/*.dom "$out/cwd-driver" > "$out/cwd-inputs.sha256"
