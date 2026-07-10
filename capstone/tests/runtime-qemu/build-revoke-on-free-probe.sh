#!/usr/bin/env bash
set -euo pipefail

# Build the controller and domain payloads for the Phase-0 revoke-on-free
# allocator probe (task 008). Two toolchains, as in
# build-intra-domain-mrev-revoke-probe.sh: the controller is Buildroot gcc, the
# domain payloads are domain_main .dom images built with the Capstone clang.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/revoke-on-free-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

PROBES=(alloc_use_after_free_fault alloc_no_free_ok alloc_sibling_survives_ok)

mkdir -p "$TMP_ROOT" "$OUT_DIR"

"$GUEST_CC" \
  -O2 \
  -I"$PROBE_DIR" \
  -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/revoke_on_free_probe.user" \
  "$PROBE_DIR/revoke_on_free_probe_guest.c" \
  "$LIBCAPSTONE_C"
printf 'Built %s\n' "$OUT_DIR/revoke_on_free_probe.user"

ASM_DIR="$OUT_DIR/asm"
mkdir -p "$ASM_DIR"

for p in "${PROBES[@]}"; do
  "$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding \
    "$DOMAIN_OPT_LEVEL" -S "$PROBE_DIR/$p.c" -o "$ASM_DIR/$p.s"
  DOMAIN_OPT_LEVEL="$DOMAIN_OPT_LEVEL" \
    bash "$SCRIPT_DIR/build-domain.sh" "$PROBE_DIR/$p.c" "$OUT_DIR/$p.dom" >/dev/null
  printf 'Built %s (%s)\n' "$OUT_DIR/$p.dom" "$DOMAIN_OPT_LEVEL"
done

printf 'asm: %s\n' "$ASM_DIR"
