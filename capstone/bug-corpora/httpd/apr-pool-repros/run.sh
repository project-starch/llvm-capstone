#!/usr/bin/env bash
# Build against upstream's apr_pools.c, byte for byte, through the census seam,
# and run the control arm first. A failed control exits 75 with no verdict.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CENSUS=${APR_CENSUS_DIR:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/apr-census}
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/apr-pool-repros}
mkdir -p "$OUT"
if [ ! -f "$CENSUS/apr_pools.c" ]; then
  echo "CONTROL-FAILED no census output at $CENSUS (bash ../../ports/apr/build-apr-census.sh)" >&2
  exit 75
fi
"${CC:-cc}" -O1 -g -o "$OUT/defects" "$HERE/shared/defects.c" "$CENSUS/apr_pools.c" \
  "$HERE/shared/stubs.c" -I"$CENSUS" || { echo "CONTROL-FAILED build" >&2; exit 75; }
fixed=$("$OUT/defects" fixed) || { echo "CONTROL-FAILED fixed arm" >&2; exit 75; }
grep -q '^VERDICT FIXED' <<<"$fixed" || { echo "CONTROL-FAILED handle not cleared" >&2; exit 75; }
buggy=$("$OUT/defects") || { echo "CONTROL-FAILED buggy arm" >&2; exit 75; }
printf '%s\n%s\n' "$fixed" "$buggy" | sed 's/^/9e6be73065 /'
grep -q '^VERDICT DEFECT-REPRODUCED' <<<"$buggy"
