#!/usr/bin/env bash
# The native-detect arm: every case under AddressSanitizer, the allocator port included.
#
#   bash runners/run-asan.sh <fresh outdir>
#
# The native port carves slab pages AND cache.c objects out of ONE payload block the driver takes
# with aligned_alloc (MCP_PAYLOAD_BYTES, 64 MiB; src/native/authority.c), and recycles them on its
# own free lists, so neither a crossing nor a stale access ever reaches a redzone or free(). The
# controls make that a reading: in the same build ASan must report one byte past, and a read after
# free of, a block of exactly that size.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
TOOLS=$(cd -- "$ROOT/../../tools" && pwd)
OUT=${1:?usage: run-asan.sh <fresh outdir>}
[ -e "$OUT" ] && { echo "CONTROL-FAILED $OUT exists" >&2; exit 75; }
SAN="-fsanitize=address -fno-omit-frame-pointer"
# build-cases.sh runs "${CC:-cc}" QUOTED, so a CC that carries flags is one word that names no
# command; a wrapper carries the sanitizer instead. It appends -O1 -g after the compiler, so the
# case objects are -O1 here; the port library takes the same sanitizer through CMAKE_C_FLAGS.
mkdir -p "$OUT"
printf '#!/bin/sh\nexec %s %s "$@"\n' "${CC:-cc}" "$SAN" > "$OUT/cc-asan"; chmod +x "$OUT/cc-asan"
CC="$OUT/cc-asan" bash "$ROOT/shared/build-cases.sh" native "$OUT" \
  -DCMAKE_C_FLAGS="$SAN -g" -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address > "$OUT.build.log" 2>&1 \
  || { echo "CONTROL-FAILED ASan build (see $OUT.build.log)" >&2; exit 75; }
${CC:-cc} $SAN -g -O0 -o "$OUT/bin/asan-control" "$TOOLS/asan-control.c" || exit 75
python3 "$TOOLS/run-native-asan.py" --corpus "$ROOT" --bin "$OUT/bin/defect-{nn}" --out "$OUT/run" \
  --build-note "$(${CC:-cc} --version | head -1); $SAN -O1 -g; port library with the same sanitizer" \
  --control "$OUT/bin/asan-control past 67108864=heap-buffer-overflow" \
  --control "$OUT/bin/asan-control uaf 67108864=heap-use-after-free"
