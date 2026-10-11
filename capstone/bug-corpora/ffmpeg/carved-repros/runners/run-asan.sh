#!/usr/bin/env bash
# The native-detect arm: every case under AddressSanitizer, fixed then buggy.
#
#   bash runners/run-asan.sh <fresh outdir>
#
# Each case's carved block is ONE calloc/malloc, and the crossing stays inside it, so ASan's
# redzones -- which sit around an allocation, never inside one -- have nothing to report. The
# controls make that a reading rather than an assumption: in the same build ASan must report a read
# one byte past a block the size of this corpus's largest carved block (case 0's edge-emulation
# buffer, 34816 bytes) and of its smallest (case 4's, 132), and a read after free.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
TOOLS=$(cd -- "$ROOT/../../tools" && pwd)
OUT=${1:?usage: run-asan.sh <fresh outdir>}
[ -e "$OUT" ] && { echo "CONTROL-FAILED $OUT exists" >&2; exit 75; }
mkdir -p "$OUT/bin"
CC=${CC:-cc}
SAN="-fsanitize=address -fno-omit-frame-pointer -g -O0"
for dir in "$ROOT"/[0-9][0-9]_*/; do
  name=$(basename "$dir")
  $CC $SAN -o "$OUT/bin/$name" "$dir/case.c" "$ROOT/shared/driver.c" -I"$ROOT/shared" \
    || { echo "CONTROL-FAILED build $name" >&2; exit 75; }
done
$CC $SAN -o "$OUT/bin/asan-control" "$TOOLS/asan-control.c" || exit 75
python3 "$TOOLS/run-native-asan.py" --corpus "$ROOT" --bin "$OUT/bin/{name}" --out "$OUT/run" \
  --build-note "$($CC --version | head -1); $SAN" \
  --control "$OUT/bin/asan-control past 34816=heap-buffer-overflow" \
  --control "$OUT/bin/asan-control past 132=heap-buffer-overflow" \
  --control "$OUT/bin/asan-control past 12288=heap-buffer-overflow" \
  --control "$OUT/bin/asan-control uaf 34816=heap-use-after-free"
