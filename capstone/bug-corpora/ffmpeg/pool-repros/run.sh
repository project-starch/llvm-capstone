#!/usr/bin/env bash
# Build the case against the port's native pool library and run its control
# first. A failed control exits 75 with no verdict, as the other corpora do.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PORT=$HERE/../../../ports/ffmpeg/buffer-pool
BUILD=${FFPOOL_NATIVE_BUILD_DIR:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ffmpeg-buffer-pool/build/native}
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ffmpeg-pool-repros}
mkdir -p "$OUT"

if [ ! -f "$BUILD/libffmpeg-pool.a" ]; then
  echo "CONTROL-FAILED no native port build at $BUILD (cmake --preset native && cmake --build --preset native)" >&2
  exit 75
fi
"${CC:-cc}" -O1 -g -o "$OUT/defects" "$HERE/shared/defects.c" \
  -I"$PORT/src/shared" -I"$BUILD/sources/ffmpeg-ported" -I"$PORT/cmake/replay-config" \
  -I"$HERE/../../../runtime" -L"$BUILD" -lffmpeg-pool || { echo "CONTROL-FAILED build" >&2; exit 75; }

status=0
for case in 461fb22053 1886c3269d 316531e61c a024f8c541 8061098418; do
  fixed=$("$OUT/defects" "$case" fixed); rc=$?
  [ $rc -eq 0 ] || { echo "CONTROL-FAILED $case fixed arm rc=$rc" >&2; exit 75; }
  grep -q '^VERDICT FIXED' <<<"$fixed" || { echo "CONTROL-FAILED $case fixed arm did not hold its reference" >&2; exit 75; }
  buggy=$("$OUT/defects" "$case"); rc=$?
  [ $rc -eq 0 ] || { echo "CONTROL-FAILED $case buggy arm rc=$rc" >&2; exit 75; }
  printf '%s\n%s\n' "$fixed" "$buggy" | sed "s/^/$case /"
  grep -q '^VERDICT DEFECT-REPRODUCED' <<<"$buggy" || status=1
done
exit $status
