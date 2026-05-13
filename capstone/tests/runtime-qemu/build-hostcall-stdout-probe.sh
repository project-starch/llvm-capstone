#!/usr/bin/env bash
set -euo pipefail

# This wrapper builds a guest-side diagnostic, not a native EM_CAPSTONE sample.
# It intentionally uses the Buildroot RISC-V guest compiler because both the
# Linux helper and the custom .smode payload execute inside the current guest
# runtime environment rather than through the LLVM Capstone backend toolchain.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/hostcall-stdout-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

mkdir -p "$TMP_ROOT" "$OUT_DIR"

# Build the ordinary Linux userspace helper that drives the two-round protocol.
"$GUEST_CC" \
  -O2 \
  -I"$PROBE_DIR" \
  -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/hostcall_stdout_probe.user" \
  "$PROBE_DIR/hostcall_stdout_probe_guest.c" \
  "$LIBCAPSTONE_C"

# Build the minimal S-mode payload that speaks the shared-memory HostCall ABI.
"$GUEST_CC" \
  -static -nostdlib -fPIC \
  -I"$PROBE_DIR" \
  -o "$OUT_DIR/hostcall_stdout_probe.smode" \
  "$PROBE_DIR/hostcall_stdout_probe.smode.c"

printf 'Built %s\n' "$OUT_DIR/hostcall_stdout_probe.user"
printf 'Built %s\n' "$OUT_DIR/hostcall_stdout_probe.smode"


