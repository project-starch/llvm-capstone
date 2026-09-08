#!/usr/bin/env bash
# Build the transfer-annotated silicon probe (Phase B item 2): the freestanding board host `rtpc`
# (buildroot gcc, no libc, integer-only output -- the borrow_cost_fpga_nogp_ctl recipe) and the
# domain `rev_transferred_probe.dom` in the SILICON config through build-ladder-domain.sh with the
# INTERP glue (DOMAIN_GLUE=interp: globals survive re-entry; the generated glue rebuilds them and
# would lose the parked capability between the two calls). Entry VA 0xA0000 so it collides with no
# staged rung (k800 0x20000, BEEBS 0x30000-0x90000, SLT 0x10000).
#   usage: build-rev-transferred-probe-fpga.sh [OUT_DIR]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../capstone-test-env.sh" >/dev/null 2>&1 || true
OUT_DIR=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/rev-transferred-probe}
BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR:-$SCRIPT_DIR/../../caplifive-buildroot}
GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LADDER_DIR="$SCRIPT_DIR/../runtime-qemu/silicon-ladder"
mkdir -p "$OUT_DIR"
"$GUEST_CC" \
  -Os -static -no-pie -fno-pie -nostdlib -ffreestanding -fno-stack-protector \
  -march=rv64imac -mabi=lp64 \
  -o "$OUT_DIR/rtpc" "$SCRIPT_DIR/rev_transferred_probe_ctl.c"
printf 'Built %s\n' "$OUT_DIR/rtpc"
DOMAIN_GLUE=interp DOMAIN_BASE_VA=${DOMAIN_BASE_VA:-0xA0000} \
  bash "$LADDER_DIR/build-ladder-domain.sh" "$LADDER_DIR/rev_transferred_probe_fpga_app.c" "$OUT_DIR/rev_transferred_probe.dom"
printf 'Built %s\n' "$OUT_DIR/rev_transferred_probe.dom"
