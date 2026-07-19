#!/usr/bin/env bash
set -euo pipefail

# Build the RTL/FPGA borrow-cost variant (task-016): a Linux-userspace controller
# (.user, buildroot gcc) + a Capstone domain payload (.dom, Capstone clang). Same
# two-toolchain split as build-borrow-cost-probe.sh; only the sources differ (the
# hardware variant reads rdcycle and hands results back through the region).
#
# Output: $OUT_DIR/borrow_cost_fpga.user and $OUT_DIR/borrow_cost_fpga.dom.
# These two files are what get dropped into the caplifive-system FPGA rootfs
# overlay (see README.md) so they ship inside fw_payload.bin.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-rtl-smoke}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O2}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
QEMU_PROBE_DIR="$SCRIPT_DIR/../runtime-qemu/borrow-cost-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

mkdir -p "$TMP_ROOT" "$OUT_DIR"

# Controller (guest Linux userspace, glibc). NOTE: on the CapliFive captype-fixed
# bitstream this .user HANGS -- the core rejects glibc's hard-float `fsd` (double
# FP) even with mstatus.FS enabled (JTAG-confirmed 2026-07-20; the DT advertises
# rv64imafdc but double-FP does not execute). Kept for QEMU / D-capable cores.
"$GUEST_CC" \
  -O2 \
  -I"$SCRIPT_DIR" \
  -I"$QEMU_PROBE_DIR" \
  -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/borrow_cost_fpga.user" \
  "$SCRIPT_DIR/borrow_cost_probe_guest_fpga.c" \
  "$LIBCAPSTONE_C"
printf 'Built %s\n' "$OUT_DIR/borrow_cost_fpga.user"

# Freestanding soft-float controller -- the one that actually runs on the
# captype-fixed silicon. -nostdlib -static (no glibc, no ld.so) + soft-float
# (-march=rv64imac -mabi=lp64) => emits ZERO FP instructions, so it never hits
# the fsd the core rejects. Own _start (inits sp + gp), raw Linux syscalls via
# ecall, integer-only output. Same ioctl protocol as libcapstone. See
# borrow_cost_fpga_ctl.c and agent-handoff/history for the full root-cause trail.
"$GUEST_CC" \
  -Os -static -no-pie -fno-pie -nostdlib -ffreestanding -fno-stack-protector \
  -march=rv64imac -mabi=lp64 \
  -I"$SCRIPT_DIR" \
  -I"$QEMU_PROBE_DIR" \
  -o "$OUT_DIR/borrow_cost_fpga_ctl" \
  "$SCRIPT_DIR/borrow_cost_fpga_ctl.c"
printf 'Built %s\n' "$OUT_DIR/borrow_cost_fpga_ctl"

# Domain payload (Capstone clang). Emit -O2 asm alongside for a static cross-check
# of the per-op cycle counts, same as the QEMU probe.
ASM_DIR="$OUT_DIR/asm"
mkdir -p "$ASM_DIR"
"$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding \
  "$DOMAIN_OPT_LEVEL" -I"$SCRIPT_DIR" -I"$QEMU_PROBE_DIR" \
  -S "$SCRIPT_DIR/borrow_cost_fpga.c" -o "$ASM_DIR/borrow_cost_fpga.s"

DOMAIN_OPT_LEVEL="$DOMAIN_OPT_LEVEL" \
  bash "$SCRIPT_DIR/../runtime-qemu/build-domain.sh" \
  "$SCRIPT_DIR/borrow_cost_fpga.c" "$OUT_DIR/borrow_cost_fpga.dom" >/dev/null
printf 'Built %s (%s)\n' "$OUT_DIR/borrow_cost_fpga.dom" "$DOMAIN_OPT_LEVEL"
printf 'asm: %s\n' "$ASM_DIR/borrow_cost_fpga.s"
