#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?output directory}
: "${CAPSTONE_EXCHANGE_CC:?exchange SDK compiler}" "${CAPSTONE_NATIVE_CC:?direct IO SDK compiler}"
mkdir -p "$out"
"$CAPSTONE_EXCHANGE_CC" -O2 -Wall -Wextra "$here/directory-services.c" -o "$out/directory-services-exchange.dom"
"$CAPSTONE_NATIVE_CC" -O2 -Wall -Wextra "$here/directory-services.c" -o "$out/directory-services-direct.dom"
sha256sum "$here/directory-services.c" "$here/build-directory.sh" \
    "$out/directory-services-exchange.dom" "$out/directory-services-direct.dom" > "$out/directory-inputs.sha256"
