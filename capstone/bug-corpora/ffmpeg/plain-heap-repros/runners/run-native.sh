#!/usr/bin/env bash
# One program per case, built twice: plain, and with AddressSanitizer.
#
#   plain  gives the fix differential -- the buggy arm's index leaves the
#          allocation, the fixed arm's does not. The control (fixed) arm runs
#          FIRST; if it does not hold, nothing below is a verdict.
#   asan   gives native-detect, two-sided. Unlike this tree's sub-object corpora,
#          these defects cross the malloc bound ITSELF, so ASan DOES see them --
#          and the fixed arm must stay silent, or the detector proves nothing.
#
# The ASan check is keyed on `heap-buffer-overflow`, never on the string
# "AddressSanitizer": LeakSanitizer's summary line contains that string too, so a
# leak would read as a bounds detection on every arm. That inverted a reading in
# the sibling sub-object corpus on 2026-10-06.
#
# An infrastructure failure exits 75 and prints no verdict.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ff-plain-heap-repros}
mkdir -p "$OUT"
CC=${CC:-cc}

status=0
for dir in "$ROOT"/[0-9][0-9]_*; do
  name=$(basename "$dir")
  n=${name%%_*}; n=${n#0}; n=${n:-0}

  "$CC" -O1 -g -o "$OUT/$name" "$dir/case.c" "$ROOT/shared/driver.c" -I"$ROOT/shared" -lm \
    || { echo "CONTROL-FAILED build $name" >&2; exit 75; }
  # -O0 for the sanitiser build: at -O1 a discarded out-of-bounds read can be
  # optimised away, which would make a silence an artefact of the compiler.
  "$CC" -O0 -g -fsanitize=address -o "$OUT/$name.asan" "$dir/case.c" "$ROOT/shared/driver.c" \
    -I"$ROOT/shared" -lm || { echo "CONTROL-FAILED asan build $name" >&2; exit 75; }

  # --- control arm first, plain build
  fixed_out=$("$OUT/$name" fixed "$n" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo "CONTROL-FAILED $name fixed arm rc=$rc" >&2; exit 75; fi
  case "$fixed_out" in
    *"VERDICT FIXED"*) ;;
    *) echo "CONTROL-FAILED $name control did not hold" >&2; exit 75 ;;
  esac

  buggy_out=$("$OUT/$name" buggy "$n" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo "CONTROL-FAILED $name buggy arm rc=$rc" >&2; exit 75; fi

  # --- native-detect, both directions. The buggy arm is EXPECTED to report.
  asan_fixed=$("$OUT/$name.asan" fixed "$n" 2>&1); arc=$?
  asan_buggy=$("$OUT/$name.asan" buggy "$n" 2>&1)

  fixed_clean=0
  case "$asan_fixed" in
    *heap-buffer-overflow*) ;;
    *) [ $arc -eq 0 ] && fixed_clean=1 ;;
  esac
  buggy_reported=0
  case "$asan_buggy" in *heap-buffer-overflow*) buggy_reported=1 ;; esac

  printf '%s\n%s\n' "$fixed_out" "$buggy_out" | sed "s|^|$name |"
  printf '%s asan fixed_clean=%s buggy_reported=%s\n' "$name" "$fixed_clean" "$buggy_reported"

  case "$buggy_out" in
    *"VERDICT DEFECT-REPRODUCED"*) ;;
    *) status=1 ;;
  esac
  # A detector that does not fire on the buggy arm, or that fires on the fixed
  # one, makes the native-detect cell void rather than negative.
  [ "$buggy_reported" = 1 ] && [ "$fixed_clean" = 1 ] || status=1
done
exit $status
