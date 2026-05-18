#!/usr/bin/env bash
set -euo pipefail

# Build the first handle-based FILE_READ proof: open a file through the helper,
# request bytes through the returned token, consume the response payload, then finish.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/hostcall-file-handle-read-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

mkdir -p "$TMP_ROOT" "$OUT_DIR"

"$GUEST_CC" \
  -O2 \
  -I"$PROBE_DIR" \
  -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/hostcall_file_handle_read_probe.user" \
  "$PROBE_DIR/hostcall_file_handle_read_probe_guest.c" \
  "$LIBCAPSTONE_C"

"$GUEST_CC" \
  -static -nostdlib -fPIC \
  -I"$PROBE_DIR" \
  -o "$OUT_DIR/hostcall_file_handle_read_probe.smode" \
  "$PROBE_DIR/hostcall_file_handle_read_probe.smode.c"

printf 'Built %s\n' "$OUT_DIR/hostcall_file_handle_read_probe.user"
printf 'Built %s\n' "$OUT_DIR/hostcall_file_handle_read_probe.smode"

