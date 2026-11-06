#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?output directory}
: "${CROSS_COMPILE:?Linux compiler prefix}" "${CAPSTONE_NATIVE_CC:?copyable SDK compiler}"
mkdir -p "$out"
"$CAPSTONE_NATIVE_CC" -O2 "$here/compute.c" -o "$out/compute.dom"
"${CROSS_COMPILE}gcc" -static -O2 -Wall -Wextra -Werror "$here/task-driver.c" -o "$out/task-driver"
