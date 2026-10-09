#!/usr/bin/env bash
# The native-detect arm: every case under AddressSanitizer, with the buffer-pool port's library
# ITSELF built with ASan, so a silence cannot come from an uninstrumented pool.
#
#   bash runners/run-asan.sh <fresh outdir>
#
# The pool hands out storage from the payload arena the driver takes with ONE aligned_alloc and
# recycles it without free(), so stock ASan has nothing to key on. The controls make that a reading
# rather than an assumption: in the same build, ASan must report a read one byte past a block the
# size of that arena (FF2_PAYLOAD_BYTES, 64 MiB) and a read after free of one.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
PORT=$(cd -- "$ROOT/../../../ports/ffmpeg/buffer-pool" && pwd)
TOOLS=$(cd -- "$ROOT/../../tools" && pwd)
OUT=${1:?usage: run-asan.sh <fresh outdir>}
[ -e "$OUT" ] && { echo "CONTROL-FAILED $OUT exists" >&2; exit 75; }
mkdir -p "$OUT/bin"
CC=${CC:-cc}
SAN="-fsanitize=address -fno-omit-frame-pointer -g -O0"
LIB=$OUT/port
cmake --preset native -S "$PORT" -B "$LIB" -DCMAKE_C_FLAGS="$SAN" -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address \
  > "$OUT/port.log" 2>&1 && cmake --build "$LIB" --target ffmpeg-pool -j "${JOBS:-12}" >> "$OUT/port.log" 2>&1 \
  || { echo "CONTROL-FAILED ASan port build (see $OUT/port.log)" >&2; exit 75; }
for dir in "$ROOT"/[0-9][0-9]_*/; do
  name=$(basename "$dir")
  $CC $SAN -o "$OUT/bin/$name" "$dir/case.c" "$ROOT/shared/driver.c" \
    -I"$ROOT/shared" -I"$PORT/src/shared" -I"$LIB/sources/ffmpeg-ported" \
    -I"$PORT/cmake/replay-config" -I"$ROOT/../../../runtime" -L"$LIB" -lffmpeg-pool \
    || { echo "CONTROL-FAILED build $name" >&2; exit 75; }
done
$CC $SAN -o "$OUT/bin/asan-control" "$TOOLS/asan-control.c" || exit 75
python3 "$TOOLS/run-native-asan.py" --corpus "$ROOT" --bin "$OUT/bin/{name}" --out "$OUT/run" \
  --build-note "$($CC --version | head -1); $SAN; port library built with the same flags" \
  --control "$OUT/bin/asan-control past 67108864=heap-buffer-overflow" \
  --control "$OUT/bin/asan-control uaf 67108864=heap-use-after-free"
