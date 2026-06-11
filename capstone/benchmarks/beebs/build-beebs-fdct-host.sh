#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${OUT_DIR:-$TMP_ROOT/beebs-build}

GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"

mkdir -p "$OUT_DIR"

"$GUEST_CC" \
  -O2 \
  -I"$SCRIPT_DIR" \
  -o "$OUT_DIR/beebs_fdct_host.user" \
  "$SCRIPT_DIR/beebs_fdct_host.c" \
  "$LIBCAPSTONE_C"

echo "Built $OUT_DIR/beebs_fdct_host.user"
