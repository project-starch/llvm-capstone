#!/usr/bin/env bash
# build.sh OUT: sublet-layout.c linked against slabs.c and cache.c of the allocators port's Sublet
# build (MCP_SUBLET=ON: the port's patch 0001 and this tree's application patch 0006), on the virtual
# SDK that CAPSTONE_SDK names, as bug-corpora/memcached/allocator-repros/shared/build-cases.sh builds
# that corpus's cases. Writes OUT/sublet-layout.dom, the port build in OUT/port and the sha256 of the
# patch it applied in OUT/inputs.sha256.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../../../.." && pwd)
PORT=$REPO/capstone/ports/memcached/allocators
[[ $# == 1 ]] || { echo 'usage: build.sh OUT' >&2; exit 2; }
: "${CAPSTONE_SDK:?set CAPSTONE_SDK to a virtual Capstone application SDK}"
OUT=$(realpath -m "$1")
mkdir -p "$OUT"

cmake --preset capstone-application -S "$PORT" -B "$OUT/port" -DMCP_SUBLET=ON > "$OUT/port.log" 2>&1
cmake --build "$OUT/port" >> "$OUT/port.log" 2>&1
SRC=$(ls -d "$OUT"/port/source/memcached-*)

"$CAPSTONE_SDK/capstone-cc" -std=c11 -O1 -g -Wall -Wextra -DNDEBUG -DCHUNK_ALIGN_BYTES=16 -DMC_CAPSTONE_SUBLET \
  -I"$PORT/src/shared" -I"$SRC" -I"$PORT/../adapted" -I"$REPO/capstone/runtime/include" \
  -o "$OUT/sublet-layout.dom" "$HERE/sublet-layout.c" "$OUT/port/libmemcached-allocators.a"
sha256sum "$REPO"/capstone/ports/memcached/app/patches/*-0006-*.patch "$OUT/sublet-layout.dom" > "$OUT/inputs.sha256"
echo "built $OUT/sublet-layout.dom"
