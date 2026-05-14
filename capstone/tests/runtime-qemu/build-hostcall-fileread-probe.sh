#!/usr/bin/env bash
set -euo pipefail

# This wrapper builds the first reverse-direction HostCall v0 proof on the same
# guest-side toolchain path as the earlier stdout/filewrite probes.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/hostcall-fileread-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

mkdir -p "$TMP_ROOT" "$OUT_DIR"

# Build the ordinary Linux helper that reads bytes and publishes them back to the
# domain through a borrowed input-style payload share.
"$GUEST_CC" \
  -O2 \
  -I"$PROBE_DIR" \
  -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/hostcall_fileread_probe.user" \
  "$PROBE_DIR/hostcall_fileread_probe_guest.c" \
  "$LIBCAPSTONE_C"

# Build the minimal S-mode payload that requests HC_V0_OP_READ_GUEST_TMPFILE.
"$GUEST_CC" \
  -static -nostdlib -fPIC \
  -I"$PROBE_DIR" \
  -o "$OUT_DIR/hostcall_fileread_probe.smode" \
  "$PROBE_DIR/hostcall_fileread_probe.smode.c"

printf 'Built %s\n' "$OUT_DIR/hostcall_fileread_probe.user"
printf 'Built %s\n' "$OUT_DIR/hostcall_fileread_probe.smode"

