#!/usr/bin/env bash
# Build the lifetime controls (91-93) for one virtual arm: the capstone-application branch of
# shared/build-cases.sh, copied because the controls link their own driver (driver-controls.c, run as
# `ctl-NN N VARIANT`) rather than driver.c / driver-virtual.c. The port build and every compile flag
# are build-cases.sh's, so the class table and the item layout agree with the corpus's images.
#
#   build-controls.sh OUT -DCAPSTONE_SDK=... -DCAPSTONE_LLVM_BUILD_DIR=... -DMCP_SUBLET=ON  -DMCP_STOCK_MALLOC=OFF
#   build-controls.sh OUT -DCAPSTONE_SDK=... -DCAPSTONE_LLVM_BUILD_DIR=... -DMCP_SUBLET=OFF -DMCP_STOCK_MALLOC=ON
#
# CAPSTONE_SDK must be exported too (the compile below uses its capstone-cc).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CORPUS=$(cd -- "$HERE/../.." && pwd)
REPO=$(cd -- "$CORPUS/../../../.." && pwd)
OUT=${1:?select an output directory}
shift
: "${CAPSTONE_SDK:?set CAPSTONE_SDK to a virtual Capstone application SDK}"
PORT="$REPO/capstone/ports/memcached/allocators"
mkdir -p "$OUT/bin" "$OUT/work"
work="$OUT/work/port"
cmake --preset capstone-application -S "$PORT" -B "$work" "$@" >"$OUT/work/port.log" 2>&1
cmake --build "$work" >>"$OUT/work/port.log" 2>&1
SRC=$(ls -d "$work"/source/memcached-*)
flags=()
if grep -q '^MCP_SUBLET:BOOL=ON$' "$work/CMakeCache.txt"; then
  flags=(-DMC_CARVE_BOUNDS -DMC_CAPSTONE_SUBLET)
fi
built=0
for dir in "$HERE"/[0-9][0-9]_*/; do
  number=$(basename "$dir" | cut -c1-2)
  "$CAPSTONE_SDK/capstone-cc" -std=c11 -O1 -g -Wall -Wextra -DNDEBUG -DCHUNK_ALIGN_BYTES=16 "${flags[@]}" \
    -I"$HERE" -I"$CORPUS/shared" -I"$PORT/src/shared" -I"$SRC" -I"$PORT/../adapted" \
    -I"$REPO/capstone/runtime/include" \
    -o "$OUT/bin/ctl-$number" "$dir/case.c" "$HERE/driver-controls.c" "$work/libmemcached-allocators.a"
  built=$((built + 1))
done
if [ "$built" -ne 3 ]; then
  echo "build-controls.sh: built $built controls, expected 3" >&2
  exit 2
fi
echo "built $built controls into $OUT/bin"
