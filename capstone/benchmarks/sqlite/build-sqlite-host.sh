#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/sqlite-build}
OUT_HOST=${OUT_HOST:-$OUT_DIR/sqlite_host.user}
HOST_SRC=${HOST_SRC:-$SCRIPT_DIR/sqlite_host.c}
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"

mkdir -p "$OUT_DIR"

"$GUEST_CC" -O2 -I"$SCRIPT_DIR" \
  -o "$OUT_HOST" \
  "$HOST_SRC" \
  "$LIBCAPSTONE_C"

echo "Built $OUT_HOST"
