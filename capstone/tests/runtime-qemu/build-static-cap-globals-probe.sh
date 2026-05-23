#!/usr/bin/env bash
set -euo pipefail

# Build the reduced static-capability-globals diagnostic domains:
# - one direct-use control case,
# - one file-scope static const reproducer.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-runtime-qemu-share}
PROBE_DIR="$SCRIPT_DIR/static-cap-globals-probe"
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}

mkdir -p "$TMP_ROOT" "$OUT_DIR"

bash "$SCRIPT_DIR/build-domain.sh" \
  "$PROBE_DIR/direct_use_domain.c" \
  "$OUT_DIR/static_cap_globals_direct.dom"

bash "$SCRIPT_DIR/build-domain.sh" \
  "$PROBE_DIR/static_const_domain.c" \
  "$OUT_DIR/static_cap_globals_static.dom"

"$LLVM_READOBJ" -h "$OUT_DIR/static_cap_globals_direct.dom"
"$LLVM_READOBJ" -h "$OUT_DIR/static_cap_globals_static.dom"

echo "Built $OUT_DIR/static_cap_globals_direct.dom"
echo "Built $OUT_DIR/static_cap_globals_static.dom"

