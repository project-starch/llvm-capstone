#!/usr/bin/env bash
set -euo pipefail

# Build the SQLite hierarchical (parent-close cascade) Stage-2 experiment probe.
# One engine controller (.user) + one host borrower (.smode), Buildroot guest cc.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/sqlite-hier-revoke-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

mkdir -p "$TMP_ROOT" "$OUT_DIR"

"$GUEST_CC" -O2 \
  -I"$PROBE_DIR" -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/sqlite_hier_revoke_probe.user" \
  "$PROBE_DIR/sqlite_hier_revoke_probe_guest.c" "$LIBCAPSTONE_C"

"$GUEST_CC" -static -nostdlib -fPIC \
  -I"$PROBE_DIR" \
  -o "$OUT_DIR/sqlite_hier_revoke_probe.smode" \
  "$PROBE_DIR/sqlite_hier_revoke_probe.smode.c"

echo "Built sqlite_hier_revoke_probe(.user/.smode) in $OUT_DIR"
