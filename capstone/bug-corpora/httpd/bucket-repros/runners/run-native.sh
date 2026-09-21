#!/usr/bin/env bash
# One program per case, each built from its case.c plus the shared driver and
# upstream's two allocators. The control arm runs first; an infrastructure
# failure exits 75 with no verdict.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
CENSUS=${APR_BUCKETS_CENSUS_DIR:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/apr-buckets-census}
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/apr-bucket-repros}
mkdir -p "$OUT"
if [ ! -f "$CENSUS/apr_buckets_alloc.c" ]; then
  echo "CONTROL-FAILED no census output at $CENSUS (bash ../../ports/apr/build-buckets-census.sh)" >&2
  exit 75
fi
status=0
for dir in "$ROOT"/[0-9][0-9]_*; do
  name=$(basename "$dir"); n=${name%%_*}; n=${n#0}; n=${n:-0}
  "${CC:-cc}" -O1 -g -o "$OUT/$name" "$dir/case.c" "$ROOT/shared/driver.c" \
    "$ROOT/shared/apr-stubs.c" "$ROOT/shared/count-free.c" \
    "$CENSUS/apr_buckets_alloc.c" "$CENSUS/apr_pools.c" \
    -I"$ROOT/shared" -I"$CENSUS" || { echo "CONTROL-FAILED build $name" >&2; exit 75; }
  fixed=$("$OUT/$name" fixed "$n") || { echo "CONTROL-FAILED $name fixed arm" >&2; exit 75; }
  grep -q '^VERDICT FIXED' <<<"$fixed" || { echo "CONTROL-FAILED $name control did not hold" >&2; exit 75; }
  buggy=$("$OUT/$name" buggy "$n") || { echo "CONTROL-FAILED $name buggy arm" >&2; exit 75; }
  printf '%s\n%s\n' "$fixed" "$buggy" | sed "s|^|$name |"
  grep -q '^VERDICT DEFECT-REPRODUCED' <<<"$buggy" || status=1
done
exit $status
