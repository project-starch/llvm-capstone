#!/usr/bin/env bash
set -euo pipefail

# Build the lender controller and borrower .smode payload for the M0
# borrow->revoke->use-after-revoke probe. Uses the Buildroot RISC-V guest
# compiler (this runs inside the current guest runtime world, like the
# shared-region and HostCall probes), not the in-tree Capstone compiler.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/borrow-revoke-uaf-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

mkdir -p "$TMP_ROOT" "$OUT_DIR"

# Lender / controller (ordinary guest Linux helper).
"$GUEST_CC" \
  -O2 \
  -I"$PROBE_DIR" \
  -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/borrow_revoke_uaf_probe.user" \
  "$PROBE_DIR/borrow_revoke_uaf_probe_guest.c" \
  "$LIBCAPSTONE_C"

# Borrower .smode payload.
"$GUEST_CC" \
  -static -nostdlib -fPIC \
  -I"$PROBE_DIR" \
  -o "$OUT_DIR/borrow_revoke_uaf_probe.smode" \
  "$PROBE_DIR/borrow_revoke_uaf_probe.smode.c"

printf 'Built %s\n' "$OUT_DIR/borrow_revoke_uaf_probe.user"
printf 'Built %s\n' "$OUT_DIR/borrow_revoke_uaf_probe.smode"
