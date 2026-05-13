#!/usr/bin/env bash
set -euo pipefail

# Build the guest-side helper and custom .smode payload for the shared-region
# sentinel probe. Like the HostCall probe, this uses the Buildroot RISC-V guest
# compiler rather than the in-tree Capstone compiler because it runs inside the
# current guest runtime world.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/shared-region-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

mkdir -p "$TMP_ROOT" "$OUT_DIR"

# Build the ordinary guest Linux helper.
"$GUEST_CC" \
  -O2 \
  -I"$PROBE_DIR" \
  -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/shared_region_probe.user" \
  "$PROBE_DIR/shared_region_probe_guest.c" \
  "$LIBCAPSTONE_C"

# Build the minimal S-mode payload that mutates the shared sentinel word.
"$GUEST_CC" \
  -static -nostdlib -fPIC \
  -I"$PROBE_DIR" \
  -o "$OUT_DIR/shared_region_probe.smode" \
  "$PROBE_DIR/shared_region_probe.smode.c"

printf 'Built %s\n' "$OUT_DIR/shared_region_probe.user"
printf 'Built %s\n' "$OUT_DIR/shared_region_probe.smode"

