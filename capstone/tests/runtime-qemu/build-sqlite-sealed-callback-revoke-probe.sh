#!/usr/bin/env bash
set -euo pipefail

# Build the SQLite SEALED-CALLBACK (callback-context revoke) Stage-2 feasibility
# probe. One host-binding controller (.user) + one engine callback body (.smode),
# Buildroot guest cc. Uses EXISTING ops only (shared_region_annotated + revoke_region
# + the already-sealed domain entry) -- no new firmware op.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/sqlite-sealed-callback-revoke-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

mkdir -p "$TMP_ROOT" "$OUT_DIR"

"$GUEST_CC" -O2 \
  -I"$PROBE_DIR" -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/sqlite_sealed_callback_revoke_probe.user" \
  "$PROBE_DIR/sqlite_sealed_callback_revoke_probe_guest.c" "$LIBCAPSTONE_C"

"$GUEST_CC" -static -nostdlib -fPIC \
  -I"$PROBE_DIR" \
  -o "$OUT_DIR/sqlite_sealed_callback_revoke_probe.smode" \
  "$PROBE_DIR/sqlite_sealed_callback_revoke_probe.smode.c"

echo "Built sqlite_sealed_callback_revoke_probe(.user/.smode) in $OUT_DIR"
