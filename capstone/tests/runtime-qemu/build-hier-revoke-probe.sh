#!/usr/bin/env bash
set -euo pipefail

# Build the controller and domain payloads for the Phase-0 hierarchical revoke
# probe (task 010, checkpoint H). Two toolchains, as in
# build-revoke-on-free-probe.sh: the controller is Buildroot gcc, the domain
# payloads are domain_main .dom images built with the Capstone clang.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/hier-revoke-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

PROBES=(hier_child_revoked_fault hier_no_close_ok hier_sibling_conn_survives_ok)

mkdir -p "$TMP_ROOT" "$OUT_DIR"

"$GUEST_CC" \
  -O2 \
  -I"$PROBE_DIR" \
  -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/hier_revoke_probe.user" \
  "$PROBE_DIR/hier_revoke_probe_guest.c" \
  "$LIBCAPSTONE_C"
printf 'Built %s\n' "$OUT_DIR/hier_revoke_probe.user"

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
