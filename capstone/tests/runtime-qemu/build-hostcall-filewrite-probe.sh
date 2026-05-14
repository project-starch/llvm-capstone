#!/usr/bin/env bash
set -euo pipefail

# This wrapper builds the second HostCall v0 proof on the same guest-side toolchain
# path as the stdout probe. It still targets the current guest runtime world, not a
# native EM_CAPSTONE hosted userspace flow.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/hostcall-filewrite-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

mkdir -p "$TMP_ROOT" "$OUT_DIR"

# Build the ordinary Linux helper that services the second opcode through the same
# two-round metadata ABI used by the stdout proof.
"$GUEST_CC" \
  -O2 \
  -I"$PROBE_DIR" \
  -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/hostcall_filewrite_probe.user" \
  "$PROBE_DIR/hostcall_filewrite_probe_guest.c" \
  "$LIBCAPSTONE_C"

# Build the minimal S-mode payload that requests HC_V0_OP_WRITE_GUEST_TMPFILE.
"$GUEST_CC" \
  -static -nostdlib -fPIC \
  -I"$PROBE_DIR" \
  -o "$OUT_DIR/hostcall_filewrite_probe.smode" \
  "$PROBE_DIR/hostcall_filewrite_probe.smode.c"

printf 'Built %s\n' "$OUT_DIR/hostcall_filewrite_probe.user"
printf 'Built %s\n' "$OUT_DIR/hostcall_filewrite_probe.smode"

