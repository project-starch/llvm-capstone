#!/usr/bin/env bash
# Build the Q-03 module-consistency loader (guest-Linux program, Buildroot gcc). No lock needed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../capstone-test-env.sh" >/dev/null 2>&1 || true
OUT_DIR=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/capstone-runtime-qemu-share}
BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR:-$SCRIPT_DIR/../../caplifive-buildroot}
GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_C="$BUILDROOT_DIR/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$BUILDROOT_DIR/package/modcapstone/include"
mkdir -p "$OUT_DIR"
"$GUEST_CC" -O2 -I"$MODCAPSTONE_INCLUDE" -o "$OUT_DIR/q03_region_hole_check.user" \
  "$SCRIPT_DIR/q03-region-hole-check/q03_region_hole_check_host.c" "$LIBCAPSTONE_C"
printf 'Built %s\n' "$OUT_DIR/q03_region_hole_check.user"
