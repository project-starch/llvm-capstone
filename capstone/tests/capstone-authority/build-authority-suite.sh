#!/usr/bin/env bash
set -euo pipefail

# Build the Capstone authority / provenance negative-test suite.
#
# For each domain under domains/:
#   - emit annotated assembly into <out>/asm/<name>.s   (codegen evidence)
#   - link a loadable domain into <out>/<name>.dom        (runtime evidence)
#
# Usage: build-authority-suite.sh [out_dir]
#   out_dir defaults to $CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share so the
#   .dom files land where run-domain-smoke.py / the 9p share expect them.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}
DOMAINS_DIR="$SCRIPT_DIR/domains"
ASM_DIR="$OUT_DIR/asm"
CLANG=${CLANG:-$CAPSTONE_CLANG}
# Built at -O0 on purpose: this is an authority / ISA-behaviour suite, so we want
# the source-level operations (forged derefs, OOB loads, 9th/10th stack-passed
# pointer args, spills) preserved 1:1 rather than optimised away (merged globals,
# IPCP, constant folding, DCE all defeat these probes at -O2). The Step-3 SHRINK
# narrowing is inserted in the backend and so applies at every optimisation level.
OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}

mkdir -p "$OUT_DIR" "$ASM_DIR"

shopt -s nullglob
for src in "$DOMAINS_DIR"/*.c; do
  name=$(basename "$src" .c)
  # stack_* probes exercise -capstone-shrink-stack (default off), so they must be
  # built with it on; global/heap narrowing is on by default and needs no flag.
  extra=""
  case "$name" in
    stack_*) extra="-mllvm -capstone-shrink-stack=true" ;;
    # heap free/reuse probes compile the real umm allocator into the TU; umm
    # divides (block index math), so enable the M extension for hardware mul/div
    # (the suite links no compiler-rt).
    heap_free_*|heap_coalesce*) extra="-Xclang -target-feature -Xclang +m" ;;
  esac
  # Codegen evidence: annotated assembly at the same opt level we run.
  "$CLANG" -target capstone64-unknown-elf -ffreestanding "$OPT_LEVEL" $extra \
    -S "$src" -o "$ASM_DIR/$name.s"
  # Runtime artifact: a loadable domain.
  DOMAIN_OPT_LEVEL="$OPT_LEVEL" EXTRA_CLANG_FLAGS="$extra" \
    bash "$SCRIPT_DIR/../runtime-qemu/build-domain.sh" "$src" "$OUT_DIR/$name.dom"
done

echo "Authority suite built into $OUT_DIR"
echo "  asm:  $ASM_DIR"
ls -1 "$OUT_DIR"/*.dom
