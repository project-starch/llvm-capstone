#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?output directory}
: "${CAPSTONE_EXCHANGE_CC:?baseline SDK compiler}" "${CAPSTONE_NATIVE_CC:?direct IO SDK compiler}"
mkdir -p "$out"
# Exactly one application object, linked with each SDK.
"$CAPSTONE_EXCHANGE_CC" -O2 -c "$here/musl-copy.c" -o "$out/musl-copy.o"
"$CAPSTONE_EXCHANGE_CC" -O2 "$out/musl-copy.o" -o "$out/copy-exchange.dom"
"$CAPSTONE_NATIVE_CC" -O2 "$out/musl-copy.o" -o "$out/copy-direct.dom"
sha256sum "$out/musl-copy.o" "$out"/copy-*.dom > "$out/inputs.sha256"
