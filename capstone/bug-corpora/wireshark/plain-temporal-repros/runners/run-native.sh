#!/usr/bin/env bash
# One program per case, built twice: plain, and with AddressSanitizer.
#
#   plain  gives the fix differential -- the buggy arm's stale pointer reads the marker a LATER
#          allocation wrote, the fixed arm's does not. The control arm runs FIRST; if it does not
#          hold, nothing below is a verdict.
#   asan   gives native-detect, two-sided, keyed on `heap-use-after-free`. Keying on the string
#          "AddressSanitizer" would also match LeakSanitizer's summary, so a merely leaky probe
#          would read as a detection on every arm.
#
# NOTE the two builds measure DIFFERENT things on purpose. Under ASan the freed chunk goes to
# quarantine, so the fresh allocation does NOT reuse it and the aliasing cannot happen -- the
# buggy arm aborts at the labelled probe instead, which is the detection being measured. The
# plain build is where the aliasing is observed. Neither alone is the result.
#
# An infrastructure failure exits 75 and prints no verdict.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/wst-plain-temporal-repros}
mkdir -p "$OUT"
CC=${CC:-cc}

shopt -s nullglob
dirs=("$ROOT"/[0-9][0-9]_*)
if [ ${#dirs[@]} -eq 0 ]; then
  echo "CONTROL-FAILED no case directories under $ROOT" >&2; exit 75
fi

status=0
for dir in "${dirs[@]}"; do
  name=$(basename "$dir")
  n=${name%%_*}; n=${n#0}; n=${n:-0}

  # -O0: at higher levels the compiler may fold a store into a freed object, and the aliasing the
  # case is built to observe would stop being observable for a reason unrelated to the defect.
  "$CC" -O0 -g -o "$OUT/$name" "$dir/case.c" "$ROOT/shared/driver.c" -I"$ROOT/shared" \
    || { echo "CONTROL-FAILED build $name" >&2; exit 75; }
  "$CC" -O0 -g -fsanitize=address -o "$OUT/$name.asan" "$dir/case.c" "$ROOT/shared/driver.c" \
    -I"$ROOT/shared" || { echo "CONTROL-FAILED asan build $name" >&2; exit 75; }

  # --- control arm first, plain build
  fixed_out=$("$OUT/$name" fixed "$n" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo "CONTROL-FAILED $name fixed arm rc=$rc" >&2; exit 75; fi
  case "$fixed_out" in
    *"VERDICT FIXED"*) ;;
    *) echo "CONTROL-FAILED $name control did not hold: $fixed_out" >&2; exit 75 ;;
  esac

  # The control arm held, so a failure here is the CASE not reproducing, not the
  # instrument failing. Still exit 75 -- a case whose buggy arm does not reproduce has
  # produced no verdict -- but do not label it CONTROL-FAILED, which names the wrong arm.
  buggy_out=$("$OUT/$name" buggy "$n" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then
    echo "DID-NOT-REPRODUCE $name buggy arm rc=$rc (the fixed-arm control DID hold)" >&2
    exit 75
  fi

  # --- native-detect, both directions. The buggy arm is EXPECTED to abort.
  asan_fixed=$("$OUT/$name.asan" fixed "$n" 2>&1); arc=$?
  asan_buggy=$("$OUT/$name.asan" buggy "$n" 2>&1)

  fixed_clean=0
  case "$asan_fixed" in
    *"heap-use-after-free"*|*"double-free"*|*"heap-buffer-overflow"*) ;;
    *) [ $arc -eq 0 ] && fixed_clean=1 ;;
  esac
  buggy_seen=NO-REPORT
  case "$asan_buggy" in
    *"heap-use-after-free"*) buggy_seen=heap-use-after-free ;;
    *"double-free"*)         buggy_seen=double-free ;;
  esac

  printf '%s plain=%s asan-buggy=%s asan-fixed=%s\n' "$name" \
    "$(case "$buggy_out" in *"VERDICT DEFECT-REPRODUCED"*) echo reproduced;; *) echo NO;; esac)" \
    "$buggy_seen" \
    "$([ $fixed_clean -eq 1 ] && echo silent || echo REPORTED-OR-FAILED)"

  case "$buggy_out" in *"VERDICT DEFECT-REPRODUCED"*) ;; *) status=1 ;; esac
  [ "$buggy_seen" = NO-REPORT ] && status=1
  [ $fixed_clean -eq 1 ] || status=1
done
exit $status
