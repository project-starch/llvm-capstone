#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/rv8-build}
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
HOST_SRC="$CAPSTONE_REPO_ROOT/capstone/benchmarks/beebs/beebs_simple_host.c"
mkdir -p "$OUT_DIR"
"$GUEST_CC" -O2 -I"$(dirname "$HOST_SRC")" -o "$OUT_DIR/rv8_aes_host.user" "$HOST_SRC" "$LIBCAPSTONE_C"
echo "Built $OUT_DIR/rv8_aes_host.user"
