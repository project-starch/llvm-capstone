#!/usr/bin/env bash
set -euo pipefail

# Build the controller and the domain payloads for the LINEAR (row11) and UNINIT
# (row14) corpus probes.
#
# Two toolchains, as in build-intra-domain-mrev-revoke-probe.sh:
#   - the controller is an ordinary guest-Linux helper -> Buildroot RISC-V gcc;
#   - the domain payloads need the capability builtins (mrev/revoke/init/drop)
#     and must receive the granted region AS A CAPABILITY, so they are
#     domain_main .dom images built with the in-tree Capstone clang.
#
# DOMAIN_OPT_LEVEL selects the payload optimisation level (default -O0).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
PROBE_DIR="$SCRIPT_DIR/linear-uninit-corpus-probe"
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
MODCAPSTONE_INCLUDE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include"

PROBES=(uninit_use_before_init_fault uninit_negative_offset_fault
        uninit_init_then_use_ok
        linear_drop_use_fault linear_double_drop_fault
        linear_no_drop_ok linear_drop_sibling_ok)

mkdir -p "$TMP_ROOT" "$OUT_DIR"

"$GUEST_CC" \
  -O2 \
  -I"$PROBE_DIR" \
  -I"$MODCAPSTONE_INCLUDE" \
  -o "$OUT_DIR/linear_uninit_corpus_probe.user" \
  "$PROBE_DIR/linear_uninit_corpus_probe_guest.c" \
  "$LIBCAPSTONE_C"
printf 'Built %s\n' "$OUT_DIR/linear_uninit_corpus_probe.user"

# Codegen evidence at the same opt level we run. Two things are worth reading in
# the asm: that the `revoke` result (not a re-materialised pointer) is the base
# register of the pre-init load, and that `drop` survives the optimiser.
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
