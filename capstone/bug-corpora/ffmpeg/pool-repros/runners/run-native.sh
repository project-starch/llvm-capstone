#!/usr/bin/env bash
# One program per case, each built from its own case.c plus the shared driver,
# and each run twice against the same program. The control arm runs first; an
# infrastructure failure exits 75 with no verdict rather than recording a
# failure that reads like the defect not reproducing.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
PORT=$ROOT/../../../ports/ffmpeg/buffer-pool
BUILD=${FFPOOL_NATIVE_BUILD_DIR:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ffmpeg-buffer-pool/build/native}
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ffmpeg-pool-repros}
mkdir -p "$OUT"

if [ ! -f "$BUILD/libffmpeg-pool.a" ]; then
  echo "CONTROL-FAILED no native port build at $BUILD" >&2
  exit 75
fi

status=0
for dir in "$ROOT"/[0-9][0-9]_*; do
  name=$(basename "$dir")
  n=${name%%_*}; n=${n#0}; n=${n:-0}
  "${CC:-cc}" -O1 -g -o "$OUT/$name" "$dir/case.c" "$ROOT/shared/driver.c" \
    -I"$ROOT/shared" -I"$PORT/src/shared" -I"$BUILD/sources/ffmpeg-ported" \
    -I"$PORT/cmake/replay-config" -I"$ROOT/../../../runtime" \
    -L"$BUILD" -lffmpeg-pool || { echo "CONTROL-FAILED build $name" >&2; exit 75; }

  fixed=$("$OUT/$name" fixed "$n"); rc=$?
  [ $rc -eq 0 ] || { echo "CONTROL-FAILED $name fixed arm rc=$rc" >&2; exit 75; }
  grep -q '^VERDICT FIXED' <<<"$fixed" || { echo "CONTROL-FAILED $name control did not hold" >&2; exit 75; }
  buggy=$("$OUT/$name" buggy "$n"); rc=$?
  [ $rc -eq 0 ] || { echo "CONTROL-FAILED $name buggy arm rc=$rc" >&2; exit 75; }
  printf '%s\n%s\n' "$fixed" "$buggy" | sed "s|^|$name |"
  grep -q '^VERDICT DEFECT-REPRODUCED' <<<"$buggy" || status=1
done
exit $status
