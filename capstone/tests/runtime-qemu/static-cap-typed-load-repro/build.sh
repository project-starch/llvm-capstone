#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_QEMU_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$RUNTIME_QEMU_DIR/../capstone-test-env.sh"

OUT_DIR=${1:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-typed-load-repro-share}
GEN_DIR=${GEN_DIR:-$OUT_DIR/generated}
CLANG=${CLANG:-$CAPSTONE_CLANG}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O2}

mkdir -p "$OUT_DIR" "$GEN_DIR"

bash "$RUNTIME_QEMU_DIR/build-domain.sh" \
  "$SCRIPT_DIR/fail_fn_struct_load.c" \
  "$OUT_DIR/fail_fn_struct_load.dom"

bash "$RUNTIME_QEMU_DIR/build-domain.sh" \
  "$SCRIPT_DIR/fail_str_struct_load.c" \
  "$OUT_DIR/fail_str_struct_load.dom"

bash "$RUNTIME_QEMU_DIR/build-domain.sh" \
  "$SCRIPT_DIR/fix_fn_runtime_materialize.c" \
  "$OUT_DIR/fix_fn_runtime_materialize.dom"

bash "$RUNTIME_QEMU_DIR/build-domain.sh" \
  "$SCRIPT_DIR/fix_str_runtime_materialize.c" \
  "$OUT_DIR/fix_str_runtime_materialize.dom"

bash "$RUNTIME_QEMU_DIR/build-domain.sh" \
  "$SCRIPT_DIR/descriptor_fn_runtime_materialize.c" \
  "$OUT_DIR/descriptor_fn_runtime_materialize.dom"

bash "$RUNTIME_QEMU_DIR/build-domain.sh" \
  "$SCRIPT_DIR/descriptor_str_runtime_materialize.c" \
  "$OUT_DIR/descriptor_str_runtime_materialize.dom"

"$CLANG" -target capstone64-unknown-elf -ffreestanding "$DOMAIN_OPT_LEVEL" -emit-llvm -S \
  "$SCRIPT_DIR/fail_fn_struct_load.c" \
  -o "$GEN_DIR/fail_fn_struct_load.ll"

python3 "$SCRIPT_DIR/generate_runtime_materialize_from_ir.py" \
  --ir "$GEN_DIR/fail_fn_struct_load.ll" \
  --output-c "$GEN_DIR/autogen_fn_runtime_materialize.c"

bash "$RUNTIME_QEMU_DIR/build-domain.sh" \
  "$GEN_DIR/autogen_fn_runtime_materialize.c" \
  "$OUT_DIR/autogen_fn_runtime_materialize.dom"

"$CLANG" -target capstone64-unknown-elf -ffreestanding "$DOMAIN_OPT_LEVEL" -emit-llvm -S \
  "$SCRIPT_DIR/fail_str_struct_load.c" \
  -o "$GEN_DIR/fail_str_struct_load.ll"

python3 "$SCRIPT_DIR/generate_runtime_materialize_from_ir.py" \
  --ir "$GEN_DIR/fail_str_struct_load.ll" \
  --output-c "$GEN_DIR/autogen_str_runtime_materialize.c"

bash "$RUNTIME_QEMU_DIR/build-domain.sh" \
  "$GEN_DIR/autogen_str_runtime_materialize.c" \
  "$OUT_DIR/autogen_str_runtime_materialize.dom"

bash "$RUNTIME_QEMU_DIR/build-domain.sh" \
  "$SCRIPT_DIR/consume_emitted_gct_string_domain.c" \
  "$OUT_DIR/consume_emitted_gct_string_domain.dom"

# Array-shaped (dtoa `nums[]`-style) cases: the static-initialized array faults,
# the runtime in-place materialization (constructor-codegen pattern) works.
bash "$RUNTIME_QEMU_DIR/build-domain.sh" \
  "$SCRIPT_DIR/fail_str_array_load.c" \
  "$OUT_DIR/fail_str_array_load.dom"

bash "$RUNTIME_QEMU_DIR/build-domain.sh" \
  "$SCRIPT_DIR/fix_str_array_runtime_materialize.c" \
  "$OUT_DIR/fix_str_array_runtime_materialize.dom"

printf 'Built standalone typed-load repro domains in %s\n' "$OUT_DIR"



