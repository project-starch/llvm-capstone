#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?output directory}
: "${CAPSTONE_EXCHANGE_CC:?exchange SDK compiler}" "${CAPSTONE_NATIVE_CC:?direct IO SDK compiler}" "${CROSS_COMPILE:?Linux compiler prefix}"
mkdir -p "$out"
"$CAPSTONE_EXCHANGE_CC" -O2 -Wall -Wextra "$here/local-tmp.c" -o "$out/local-tmp-exchange.dom"
"$CAPSTONE_NATIVE_CC" -O2 -Wall -Wextra "$here/local-tmp.c" -o "$out/local-tmp-direct.dom"
"${CROSS_COMPILE}gcc" -static -O2 -Wall -Wextra "$here/local-tmp-driver.c" -o "$out/local-tmp-driver"
sha256sum "$here/local-tmp.c" "$here/local-tmp-driver.c" "$here/build-local-tmp.sh" \
    "$out/local-tmp-exchange.dom" "$out/local-tmp-direct.dom" "$out/local-tmp-driver" > "$out/local-tmp-inputs.sha256"
