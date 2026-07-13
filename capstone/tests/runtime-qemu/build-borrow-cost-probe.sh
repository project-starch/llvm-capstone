#!/usr/bin/env bash
set -euo pipefail

# Build the controller and domain payload for the borrow-path cost measurement
# (task-014, paper deliverable 2). Two toolchains, as in the intra-domain probe:
#   - controller: ordinary guest-Linux helper -> Buildroot RISC-V gcc;
#   - domain payload: needs the capability builtins (mrev/delin/revoke), so it
#     is a domain_main .dom built with the in-tree Capstone clang via
#     build-domain.sh.
#
# The domain is built at -O2 (the level the paper's overhead numbers are quoted
# at) and its -O2 assembly is emitted alongside, so the per-operation
# instruction counts can be cross-checked statically against the dynamic
# csrdicount readout.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O2}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/borrow-cost-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

mkdir -p "$TMP_ROOT" "$OUT_DIR"

"$GUEST_CC" \
  -O2 \
  -I"$PROBE_DIR" \
  -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/borrow_cost_probe.user" \
  "$PROBE_DIR/borrow_cost_probe_guest.c" \
  "$LIBCAPSTONE_C"
printf 'Built %s\n' "$OUT_DIR/borrow_cost_probe.user"

ASM_DIR="$OUT_DIR/asm"
mkdir -p "$ASM_DIR"
"$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding \
  "$DOMAIN_OPT_LEVEL" -S "$PROBE_DIR/borrow_cost.c" -o "$ASM_DIR/borrow_cost.s"

DOMAIN_OPT_LEVEL="$DOMAIN_OPT_LEVEL" \
  bash "$SCRIPT_DIR/build-domain.sh" "$PROBE_DIR/borrow_cost.c" "$OUT_DIR/borrow_cost.dom" >/dev/null
printf 'Built %s (%s)\n' "$OUT_DIR/borrow_cost.dom" "$DOMAIN_OPT_LEVEL"
printf 'asm: %s\n' "$ASM_DIR/borrow_cost.s"
