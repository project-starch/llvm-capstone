#!/usr/bin/env bash
# Build the two LD_PRELOAD probes for CheriBSD purecap.
#   build-probes.sh OUT_DIR
# mqstat.so  exit-time epochs, jemalloc ledger, kernel sweep counters
# mqtrace.so sampled allocator state, reuse distance, asked bytes (see mqtrace.c)
set -euo pipefail
O=$1
SDK=${SDK:-$HOME/cheri/output/sdk}
CC="$SDK/bin/clang --config $SDK/bin/cheribsd-riscv64-purecap.cfg -O2 -Wall -Wextra -Wno-unused-parameter -shared -fPIC"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$O"
$CC -o "$O/mqstat.so" "$HERE/mqstat.c"
$CC -o "$O/mqtrace.so" "$HERE/mqtrace.c"
echo "built $O/mqstat.so $O/mqtrace.so"
