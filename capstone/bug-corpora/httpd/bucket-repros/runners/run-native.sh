#!/usr/bin/env bash
# The native arms through the port: one program per case, each run twice,
# `fixed` and `buggy`. The control arm runs first; an infrastructure failure
# exits 75 with no verdict. Builds via shared/build-cases.sh native.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/apr-bucket-repros/native}
bash "$ROOT/shared/build-cases.sh" native "$OUT" >/dev/null || { echo "CONTROL-FAILED build" >&2; exit 75; }
status=0
for dir in "$ROOT"/[0-9][0-9]_*; do
  name=$(basename "$dir"); number=${name:0:2}; n=$((10#$number))
  bin="$OUT/bin/defect-$number"
  fixed=$("$bin" fixed "$n") || { echo "CONTROL-FAILED $name fixed arm" >&2; exit 75; }
  grep -q '^VERDICT FIXED' <<<"$fixed" || { echo "CONTROL-FAILED $name control did not hold" >&2; exit 75; }
  buggy=$("$bin" buggy "$n") || { echo "CONTROL-FAILED $name buggy arm" >&2; exit 75; }
  printf '%s\n%s\n' "$fixed" "$buggy" | sed "s|^|$name |"
  grep -q '^VERDICT DEFECT-REPRODUCED' <<<"$buggy" || status=1
done
exit $status
