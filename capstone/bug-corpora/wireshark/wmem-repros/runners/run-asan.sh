#!/usr/bin/env bash
# The native-detect arm: every case under AddressSanitizer, wmem itself included, buggy and fixed.
#
#   bash runners/run-asan.sh <fresh outdir>
#
# wmem's BLOCK allocator carves every chunk out of blocks inside ONE payload the driver takes with
# aligned_alloc (WM_PAYLOAD_BYTES, 384 MiB), and a pool reset or an individual free hands chunks
# back to wmem, never to free(). The controls make the silence a reading: in the same build ASan
# must report one byte past, and a read after free of, a block of exactly that size. That block is
# larger than ASan's default 256 MiB quarantine, which unmaps it at once on free so the stale read
# SEGVs unreported (first run, 2026-10-09); the quarantine is raised so the control can report.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
TOOLS=$(cd -- "$ROOT/../../tools" && pwd)
OUT=${1:?usage: run-asan.sh <fresh outdir>}
SAN="-fsanitize=address -fno-omit-frame-pointer -g -O0"
bash "$HERE/run-native.sh" "$OUT" -DCMAKE_C_FLAGS="$SAN" -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address > "$OUT.native.log" 2>&1
rc=$?; [ $rc -eq 75 ] && { echo "CONTROL-FAILED ASan build (see $OUT.native.log)" >&2; exit 75; }
${CC:-cc} $SAN -o "$OUT/bin/asan-control" "$TOOLS/asan-control.c" || exit 75
python3 "$TOOLS/run-native-asan.py" --corpus "$ROOT" --bin "$OUT/bin/case-{nn}" --out "$OUT/run" \
  --buggy-args "0 {n} buggy" --fixed-args "0 {n} fixed" \
  --asan-options quarantine_size_mb=1024 \
  --build-note "$(${CC:-cc} --version | head -1); $SAN; wmem and the port built with the same flags" \
  --control "$OUT/bin/asan-control past 402653184=heap-buffer-overflow" \
  --control "$OUT/bin/asan-control uaf 402653184=heap-use-after-free"
