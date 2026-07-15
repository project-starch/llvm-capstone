#!/usr/bin/env bash
set -euo pipefail

# Build the RTL/FPGA temporal-safety overhead variant (task-016): a Linux-
# userspace controller (.user, buildroot gcc) + three Capstone-clang domain
# payloads, one per allocator config (-DROF_COST_MODE: bump/norevoke/revoke).
# Same two-toolchain split as build-borrow-cost-fpga.sh; the hardware variant
# reads mcycle (fpga_instrument.h) and hands the 4 result counters back through a
# retained region, instead of the emulator-only csdebugcount ops.
#
# Output in $OUT_DIR: revoke_cost_fpga.user,
#   revoke_cost_fpga_{bump,norevoke,revoke}.dom.
# These are what get dropped into the caplifive-system FPGA rootfs overlay (see
# README.md) so they ship inside fw_payload.bin.
#
# Env:
#   DOMAIN_OPT_LEVEL       (default -O2)
#   FPGA_CYCLE_USE_RDCYCLE=1  build the domains to read rdcycle instead of mcycle
#                             (see fpga_instrument.h; default is mcycle, which is
#                             what the board needs -- user `cycle` is gated there).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-rtl-smoke}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O2}

CYCLE_FLAG=""
if [[ "${FPGA_CYCLE_USE_RDCYCLE:-0}" == "1" ]]; then
  CYCLE_FLAG="-DFPGA_CYCLE_USE_RDCYCLE"
fi

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
QEMU_PROBE_DIR="$SCRIPT_DIR/../runtime-qemu/revoke-cost-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

mkdir -p "$TMP_ROOT" "$OUT_DIR"

# Controller (guest Linux userspace).
"$GUEST_CC" \
  -O2 \
  -I"$SCRIPT_DIR" \
  -I"$QEMU_PROBE_DIR" \
  -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/revoke_cost_fpga.user" \
  "$SCRIPT_DIR/revoke_cost_probe_guest_fpga.c" \
  "$LIBCAPSTONE_C"
printf 'Built %s\n' "$OUT_DIR/revoke_cost_fpga.user"

# Domain payloads, one per allocator config. Emit -O2 asm alongside for a static
# cross-check of the per-op cycle counts, same as the QEMU probe.
ASM_DIR="$OUT_DIR/asm"
mkdir -p "$ASM_DIR"

build_mode() { # $1=name  $2=ROF_COST_MODE value
  local name="$1" mode="$2"
  "$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding \
    "$DOMAIN_OPT_LEVEL" -DROF_COST_MODE="$mode" $CYCLE_FLAG \
    -I"$SCRIPT_DIR" -I"$QEMU_PROBE_DIR" \
    -S "$SCRIPT_DIR/revoke_cost_fpga.c" -o "$ASM_DIR/revoke_cost_fpga_$name.s"
  DOMAIN_OPT_LEVEL="$DOMAIN_OPT_LEVEL" \
    EXTRA_CLANG_FLAGS="-DROF_COST_MODE=$mode $CYCLE_FLAG" \
    bash "$SCRIPT_DIR/../runtime-qemu/build-domain.sh" \
    "$SCRIPT_DIR/revoke_cost_fpga.c" "$OUT_DIR/revoke_cost_fpga_$name.dom" >/dev/null
  printf 'Built %s (%s, mode %s)\n' \
    "$OUT_DIR/revoke_cost_fpga_$name.dom" "$DOMAIN_OPT_LEVEL" "$mode"
}

build_mode bump 0
build_mode norevoke 1
build_mode revoke 2

printf 'asm: %s\n' "$ASM_DIR"
