#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?output directory}
: "${CAPSTONE_EXCHANGE_CC:?exchange SDK compiler}" "${CAPSTONE_NATIVE_CC:?direct SDK compiler}"
mkdir -p "$out"
"$CAPSTONE_EXCHANGE_CC" -O2 "$here/clock.c" -o "$out/clock-exchange.dom"
"$CAPSTONE_NATIVE_CC" -O2 "$here/clock.c" -o "$out/clock-libc.dom"
"$CAPSTONE_NATIVE_CC" -O2 -DTIME_RAW "$here/clock.c" -o "$out/clock-direct.dom"
sha256sum "$here/clock.c" "$out"/clock-*.dom > "$out/clock-inputs.sha256"
