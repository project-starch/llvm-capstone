#!/usr/bin/env bash
set -euo pipefail

# Build the revocation enforcement test-matrix probe (cases 2 and 3). Uses the
# Buildroot RISC-V guest compiler (runs in the current guest runtime world).
# One controller; one .smode per case (selected by REVOKE_MATRIX_CASE).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/revoke-matrix-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

mkdir -p "$TMP_ROOT" "$OUT_DIR"

for CASE in 2 3; do
  # Controller (one per case so REVOKE_MATRIX_CASE prints correctly).
  "$GUEST_CC" -O2 -DREVOKE_MATRIX_CASE="$CASE" \
    -I"$PROBE_DIR" -I"$MODCAPSTONE_INCLUDE" \
    -o "$OUT_DIR/revoke_matrix_probe_case$CASE.user" \
    "$PROBE_DIR/revoke_matrix_probe_guest.c" "$LIBCAPSTONE_C"
  # Borrower .smode for this case.
  "$GUEST_CC" -static -nostdlib -fPIC -DREVOKE_MATRIX_CASE="$CASE" \
    -I"$PROBE_DIR" \
    -o "$OUT_DIR/revoke_matrix_probe_case$CASE.smode" \
    "$PROBE_DIR/revoke_matrix_probe.smode.c"
  printf 'Built case %d: %s(.user/.smode)\n' "$CASE" \
    "$OUT_DIR/revoke_matrix_probe_case$CASE"
done
