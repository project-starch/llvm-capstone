#!/usr/bin/env bash
set -euo pipefail

# Build the controller and the domain payloads for the single-domain held-cap
# BORROW-REVOKE probe (row3 Option B).
#
# Two toolchains, on purpose:
#   - the controller is an ordinary guest-Linux helper -> Buildroot RISC-V gcc,
#     like every other runtime-qemu probe controller;
#   - the domain payloads need the capability builtins (mrev/delin/revoke) and
#     must receive the granted region AS A CAPABILITY, so they are domain_main
#     .dom images built with the in-tree Capstone clang via build-domain.sh.
#
# That is also why these payloads are not .smode files: a .smode payload runs
# under the sbi.dom scaffold, where a shared region lands in the scaffold's
# regions[] and S-mode reaches the bytes through ambient cpmp authority. Only the
# domain_main path (my_first_domain/start.S) hands the capability itself to
# domain C. See intra-domain-mrev-revoke-probe/probe_domain.h.
#
# DOMAIN_OPT_LEVEL selects the payload optimisation level (default -O0).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/intra-domain-mrev-revoke-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

PROBES=(held_revoke_fault held_no_revoke_ok held_mem_alias_fault
        held_unrelated_ok held_ambient_miss held_split_sibling_ok
        held_protected_value_lifecycle held_arena_survives_revoke)

mkdir -p "$TMP_ROOT" "$OUT_DIR"

"$GUEST_CC" \
  -O2 \
  -I"$PROBE_DIR" \
  -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/intra_domain_mrev_revoke_probe.user" \
  "$PROBE_DIR/intra_domain_mrev_revoke_probe_guest.c" \
  "$LIBCAPSTONE_C"
printf 'Built %s\n' "$OUT_DIR/intra_domain_mrev_revoke_probe.user"

# Codegen evidence at the same opt level we run: the asm shows whether the
# post-revoke deref stays based on the held capability register.
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
