#!/usr/bin/env bash
# The native arms: one program per case, built through the port and run twice
# against the same binary. The control arm (fixed) runs first and must hold; an
# infrastructure failure exits 75 with no verdict rather than recording a
# failure that reads like the defect not reproducing.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/memcached-allocator-repros/native}
bash "$ROOT/shared/build-cases.sh" native "$OUT" >/dev/null \
  || { echo "CONTROL-FAILED build (see $OUT/work/port.log)" >&2; exit 75; }
status=0
for dir in "$ROOT"/[0-9][0-9]_*; do
  name=$(basename "$dir"); n=${name%%_*}; bin="$OUT/bin/defect-$n"; n=${n#0}; n=${n:-0}
  fixed=$("$bin" fixed "$n") || { echo "CONTROL-FAILED $name fixed arm" >&2; exit 75; }
  grep -q '^VERDICT FIXED' <<<"$fixed" || { echo "CONTROL-FAILED $name control did not hold" >&2; exit 75; }
  buggy=$("$bin" buggy "$n") || { echo "CONTROL-FAILED $name buggy arm" >&2; exit 75; }
  printf '%s\n%s\n' "$fixed" "$buggy" | sed "s|^|$name |"
  grep -q '^VERDICT DEFECT-REPRODUCED' <<<"$buggy" || status=1
done
exit $status
