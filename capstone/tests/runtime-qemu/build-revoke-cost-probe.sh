#!/usr/bin/env bash
set -euo pipefail

# Build the temporal-safety overhead microbenchmark (Capstone side). One
# Buildroot-gcc controller + three Capstone-clang domain payloads, one per
# allocator config (-DROF_COST_MODE): bump / norevoke / revoke. Same
# two-toolchain split as build-borrow-cost-probe.sh.
#
# Output in $OUT_DIR: revoke_cost_probe.user, revoke_cost_{bump,norevoke,revoke}.dom.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O2}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/revoke-cost-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

mkdir -p "$TMP_ROOT" "$OUT_DIR"

# Controller (guest Linux userspace).
"$GUEST_CC" \
  -O2 \
  -I"$PROBE_DIR" \
  -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/revoke_cost_probe.user" \
  "$PROBE_DIR/revoke_cost_probe_guest.c" \
  "$LIBCAPSTONE_C"
printf 'Built %s\n' "$OUT_DIR/revoke_cost_probe.user"

ASM_DIR="$OUT_DIR/asm"
mkdir -p "$ASM_DIR"

# Domain payloads, one per allocator config.
build_mode() { # $1=name  $2=ROF_COST_MODE value
  local name="$1" mode="$2"
  "$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding \
    "$DOMAIN_OPT_LEVEL" -DROF_COST_MODE="$mode" \
    -I"$PROBE_DIR" -S "$PROBE_DIR/revoke_cost.c" -o "$ASM_DIR/revoke_cost_$name.s"
  DOMAIN_OPT_LEVEL="$DOMAIN_OPT_LEVEL" EXTRA_CLANG_FLAGS="-DROF_COST_MODE=$mode" \
    bash "$SCRIPT_DIR/build-domain.sh" \
    "$PROBE_DIR/revoke_cost.c" "$OUT_DIR/revoke_cost_$name.dom" >/dev/null
  printf 'Built %s (%s, mode %s)\n' "$OUT_DIR/revoke_cost_$name.dom" "$DOMAIN_OPT_LEVEL" "$mode"
}

build_mode bump 0
build_mode norevoke 1
build_mode revoke 2

printf 'asm: %s\n' "$ASM_DIR"
