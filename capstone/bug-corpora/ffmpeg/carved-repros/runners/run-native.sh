#!/usr/bin/env bash
# The native fix differential: one program per case, the control (fixed) arm first, then the
# buggy one. The buggy arm must print VERDICT DEFECT-REPRODUCED -- its access left its carved
# region -- and the fixed arm VERDICT FIXED; the driver exits 75 if a crossing leaves the
# allocation, because then the case is not nested and every bound would see it.
#
#   bash runners/run-native.sh [outdir]
#
# ASan is runners/run-asan.sh, with positive controls: its silence here is the reading.
# An infrastructure failure exits 75 and prints no verdict.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ff-carved-repros}
mkdir -p "$OUT"
CC=${CC:-cc}

status=0
for dir in "$ROOT"/[0-9][0-9]_*; do
  name=$(basename "$dir")
  n=${name%%_*}; n=${n#0}; n=${n:-0}
  "$CC" -O1 -g -Wall -Werror -o "$OUT/$name" "$dir/case.c" "$ROOT/shared/driver.c" -I"$ROOT/shared" \
    || { echo "CONTROL-FAILED build $name" >&2; exit 75; }

  fixed_out=$("$OUT/$name" fixed "$n" 2>&1); rc=$?
  [ $rc -eq 0 ] || { echo "CONTROL-FAILED $name fixed arm rc=$rc: $fixed_out" >&2; exit 75; }
  case "$fixed_out" in
    *"VERDICT FIXED"*) ;;
    *) echo "CONTROL-FAILED $name control did not hold" >&2; exit 75 ;;
  esac
  buggy_out=$("$OUT/$name" buggy "$n" 2>&1); rc=$?
  [ $rc -eq 75 ] && { echo "CONTROL-FAILED $name buggy arm: $buggy_out" >&2; exit 75; }

  printf '%s\n%s\n' "$fixed_out" "$buggy_out" | grep -v '^carve ' | sed "s|^|$name |"
  case "$buggy_out" in
    *"VERDICT DEFECT-REPRODUCED"*) [ $rc -eq 0 ] || status=1 ;;
    *) status=1 ;;
  esac
done
exit $status
