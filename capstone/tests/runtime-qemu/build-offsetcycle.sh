#!/usr/bin/env bash
# Build the offset-window probe (offsetcycle/offsetcycle_host.c) with the guest toolchain, linking the
# buildroot tree's libcapstone.c by path like every other probe.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../capstone-test-env.sh" >/dev/null 2>&1 || true
OUT_DIR=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/capstone-runtime-qemu-share}
BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR:-$SCRIPT_DIR/../../caplifive-buildroot}
GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_C="$BUILDROOT_DIR/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$BUILDROOT_DIR/package/modcapstone/include"
mkdir -p "$OUT_DIR"
"$GUEST_CC" -O2 ${OFFSETCYCLE_DEFS:-} -I"$MODCAPSTONE_INCLUDE" -o "$OUT_DIR/offsetcycle.user" \
  "$SCRIPT_DIR/offsetcycle/offsetcycle_host.c" "$LIBCAPSTONE_C"
printf 'Built %s\n' "$OUT_DIR/offsetcycle.user"
