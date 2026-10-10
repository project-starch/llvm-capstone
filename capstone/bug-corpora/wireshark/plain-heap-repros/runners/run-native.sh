#!/usr/bin/env bash
# One program per case, built twice: plain, and with AddressSanitizer.
#
#   plain  gives the fix differential -- the buggy arm's terminator address IS the
#          allocation's end, the fixed arm's is the last byte inside. The control
#          arm runs FIRST; if it does not hold, nothing below is a verdict.
#   asan   gives native-detect, two-sided. Unlike this tree's sub-object corpora,
#          these defects cross the malloc bound itself, so ASan DOES see them --
#          and the fixed arm must stay silent, or the detector proves nothing.
#
# An infrastructure failure exits 75 and prints no verdict.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ws-plain-heap-repros}
mkdir -p "$OUT"
CC=${CC:-cc}

status=0
for dir in "$ROOT"/[0-9][0-9]_*; do
  name=$(basename "$dir")
  n=${name%%_*}; n=${n#0}; n=${n:-0}

  "$CC" -O1 -g -o "$OUT/$name" "$dir/case.c" "$ROOT/shared/driver.c" -I"$ROOT/shared" \
    || { echo "CONTROL-FAILED build $name" >&2; exit 75; }
  "$CC" -O1 -g -fsanitize=address -o "$OUT/$name.asan" "$dir/case.c" "$ROOT/shared/driver.c" \
    -I"$ROOT/shared" || { echo "CONTROL-FAILED asan build $name" >&2; exit 75; }

  # --- control arm first, plain build
  fixed_out=$("$OUT/$name" fixed "$n" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo "CONTROL-FAILED $name fixed arm rc=$rc" >&2; exit 75; fi
  case "$fixed_out" in
    *"VERDICT FIXED"*) ;;
    *) echo "CONTROL-FAILED $name control did not hold" >&2; exit 75 ;;
  esac

  buggy_out=$("$OUT/$name" buggy "$n" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo "CONTROL-FAILED $name buggy arm rc=$rc" >&2; exit 75; fi

  # --- native-detect, both directions. The buggy arm is EXPECTED to abort.
  asan_fixed=$("$OUT/$name.asan" fixed "$n" 2>&1); arc=$?
  asan_buggy=$("$OUT/$name.asan" buggy "$n" 2>&1); brc=$?

  fixed_clean=0
  case "$asan_fixed" in *"AddressSanitizer"*) ;; *) [ $arc -eq 0 ] && fixed_clean=1 ;; esac
  buggy_seen=0
  case "$asan_buggy" in *"heap-buffer-overflow"*) buggy_seen=1 ;; esac

  printf '%s plain=%s asan-buggy=%s asan-fixed=%s\n' "$name" \
    "$(case "$buggy_out" in *"VERDICT DEFECT-REPRODUCED"*) echo reproduced;; *) echo NO;; esac)" \
    "$([ $buggy_seen -eq 1 ] && echo heap-buffer-overflow || echo NO-REPORT)" \
    "$([ $fixed_clean -eq 1 ] && echo silent || echo REPORTED-OR-FAILED)"

  case "$buggy_out" in *"VERDICT DEFECT-REPRODUCED"*) ;; *) status=1 ;; esac
  [ $buggy_seen -eq 1 ] || status=1
  [ $fixed_clean -eq 1 ] || status=1
done
exit $status
