#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_QEMU_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$RUNTIME_QEMU_DIR/../capstone-test-env.sh"

OUT_DIR=${1:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-typed-load-repro-share}
LLVM_OBJDUMP=${LLVM_OBJDUMP:-$CAPSTONE_LLVM_BIN/llvm-objdump}

bash "$SCRIPT_DIR/build.sh" "$OUT_DIR"

check_nonempty_gct() {
  local domain_path=$1
  local dump

  dump=$("$LLVM_OBJDUMP" -s -j .gct "$domain_path")
  printf '%s\n' "$dump"

  if ! grep -q 'SCAP' <<<"$dump"; then
    echo "expected non-empty SCAP header in .gct for $domain_path" >&2
    exit 1
  fi
}

printf '=== fail_fn_struct_load.dom .gct ===\n'
check_nonempty_gct "$OUT_DIR/fail_fn_struct_load.dom"
printf '\n=== fail_str_struct_load.dom .gct ===\n'
check_nonempty_gct "$OUT_DIR/fail_str_struct_load.dom"

echo '__STATIC_CAP_GCT_EMISSION_OK__'
printf 'Verified non-empty .gct emission for reduced static capability failing cases in %s\n' "$OUT_DIR"

