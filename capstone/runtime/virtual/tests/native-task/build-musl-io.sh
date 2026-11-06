#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?output directory}
: "${CAPSTONE_EXCHANGE_CC:?baseline SDK compiler}" "${CAPSTONE_NATIVE_CC:?direct IO SDK compiler}" "${CROSS_COMPILE:?Linux compiler prefix}"
mkdir -p "$out"
flags=(-O2 -I"$CAPSTONE_REPO_ROOT/capstone/runtime/include")
"$CAPSTONE_EXCHANGE_CC" "${flags[@]}" "$here/musl-io.c" -o "$out/io-exchange.dom"
"$CAPSTONE_NATIVE_CC" "${flags[@]}" -DMUSL_IO_DIRECT_TEST "$here/musl-io.c" -o "$out/io-direct.dom"
"${CROSS_COMPILE}gcc" -static -O2 -Wall -Wextra -Werror "$here/musl-pipe.c" -o "$out/musl-pipe"
CAPSTONE_NATIVE_STUB_IN_SDK=1 bash "$here/../native-integration/build.sh" "$out"
bash "$here/build.sh" "$out"
"$CAPSTONE_NATIVE_CC" "${flags[@]}" -DMUSL_IO_ROUTE -DCAPSTONE_RUNTIME_VIRTUAL -DCAPSTONE_NATIVE_SYSCALL_ABI \
 "$here/../native-race/race.c" "$here/../native-race/rt-stubs.S" -o "$out/race.dom"
"${CROSS_COMPILE}gcc" -static -O2 "$here/../native-race/pipe-fill.c" -o "$out/pipe-fill"
sha256sum "$out"/*.dom "$out/musl-pipe" "$out/pipe-fill" > "$out/musl-inputs.sha256"
