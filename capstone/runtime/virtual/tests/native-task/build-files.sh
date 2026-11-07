#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?output directory}
: "${CAPSTONE_EXCHANGE_CC:?exchange SDK compiler}" "${CAPSTONE_NATIVE_CC:?direct IO SDK compiler}" "${CROSS_COMPILE:?Linux compiler prefix}"
mkdir -p "$out"
"$CAPSTONE_EXCHANGE_CC" -O2 "$here/files.c" -o "$out/files-exchange.dom"
"$CAPSTONE_NATIVE_CC" -O2 "$here/files.c" -o "$out/files-native.dom"
"$CAPSTONE_NATIVE_CC" -O2 -DFILES_RAW "$here/files.c" -o "$out/files-direct.dom"
"${CROSS_COMPILE}gcc" -static -O2 -Wall -Wextra -Werror "$here/files-driver.c" -o "$out/files-driver"
sha256sum "$here/files.c" "$here/files-driver.c" "$out"/*.dom "$out/files-driver" > "$out/files-inputs.sha256"
