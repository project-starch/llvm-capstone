#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?output directory}
: "${CAPSTONE_EXCHANGE_CC:?exchange SDK compiler}" "${CAPSTONE_NATIVE_CC:?direct IO SDK compiler}"
mkdir -p "$out"
"$CAPSTONE_EXCHANGE_CC" -O2 -Wall -Wextra "$here/time-services.c" -o "$out/time-services-exchange.dom"
"$CAPSTONE_NATIVE_CC" -O2 -Wall -Wextra "$here/time-services.c" -o "$out/time-services-direct.dom"
sha256sum "$here/time-services.c" "$here/build-time.sh" \
    "$out/time-services-exchange.dom" "$out/time-services-direct.dom" > "$out/time-inputs.sha256"
