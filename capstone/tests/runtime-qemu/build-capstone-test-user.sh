#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

REPO_ROOT=${CAPSTONE_REPO_ROOT}
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
CAPSTONE_TEST_SRC=${CAPSTONE_TEST_SRC:-$REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/capstone-test.c}
LIBCAPSTONE_SRC=${LIBCAPSTONE_SRC:-$REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c}
CAPSTONE_INCLUDE_DIR=${CAPSTONE_INCLUDE_DIR:-$REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/include}

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <output-path>" >&2
  exit 1
fi

OUT=$1
mkdir -p "$(dirname -- "$OUT")"

"$GUEST_CC" -O2 -I"$CAPSTONE_INCLUDE_DIR" -o "$OUT" "$CAPSTONE_TEST_SRC" "$LIBCAPSTONE_SRC"

echo "Built $OUT"

