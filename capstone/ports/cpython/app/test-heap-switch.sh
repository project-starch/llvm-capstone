#!/bin/bash
# Test the real decision block, extracted from the script rather than copied,
# so the test cannot drift from what runs.
SRC=~/arms/cpython/pairs-worktree/capstone/ports/cpython/app/prepare-cpython-capstone.sh
BLOCK=/tmp/heap-block.sh
# From the marker comment through the fi that closes the GRANT_BYTES block.
# Counting fi does not work: the inner "if [[ \$APP_BOUNDS == 0 ]]" adds one,
# and an earlier version of this test truncated there and reported two failures
# that were its own.
awk '/^# ---- the heap this image links/{on=1} on{print} on&&/GRANT_BYTES=83886080/{want=1} want&&/^fi$/{exit}'   "$SRC" > "$BLOCK"
grep -q GRANT_BYTES "$BLOCK" || { echo "extraction missed the GRANT_BYTES block"; exit 2; }
grep -c . "$BLOCK" >/dev/null || { echo "extraction failed"; exit 2; }

run() {  # env... -> prints the resulting SDK_FLAGS, or the exit code
  ( set -euo pipefail; log(){ :; }; eval "$1"; . "$BLOCK"; printf '%s\n' "${SDK_FLAGS[*]-}" ) 2>/dev/null
  rc=$?; [ $rc -ne 0 ] && echo "EXIT=$rc"
}
pass=0; fail=0
chk() { # name expected env
  got=$(run "$3")
  if [ "$got" = "$2" ]; then echo "ok   $1"; pass=$((pass+1))
  else echo "FAIL $1"; echo "       want: [$2]"; echo "       got : [$got]"; fail=$((fail+1)); fi
}
echo "=== the property that matters: unset is the old behaviour ==="
chk "1. nothing set                -> no flags (as before)" \
    "" "true"
chk "2. CPY_SUBLET=1 alone          -> only GRANT_BYTES (as before)" \
    "-DCAPSTONE_APPLICATION_GRANT_BYTES=83886080" "CPY_SUBLET=1"
echo
echo "=== the four arms (CONTEXTS=1 on the sublet heaps is REQUIRED, not tuning:"
echo "    at the default 15, REGION_DATA is exactly the buddy allocator's"
echo "    largest block and capstone-exec cannot allocate launch regions) ==="
chk "3. CPYD_HEAP=none" \
    "-DCAPSTONE_APPLICATION_HEAP=level0 -DCMAKE_C_FLAGS_RELEASE=-O1 -DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0" \
    "CPYD_HEAP=none"
chk "4. CPYD_HEAP=bounds" \
    "-DCAPSTONE_APPLICATION_HEAP=level0" "CPYD_HEAP=bounds"
chk "5. CPYD_HEAP=sublet" \
    "-DCAPSTONE_APPLICATION_HEAP=sublet -DCAPSTONE_APPLICATION_CONTEXTS=1" "CPYD_HEAP=sublet"
chk "6. CPYD_HEAP=sublet-pymalloc" \
    "-DCAPSTONE_APPLICATION_HEAP=sublet -DCAPSTONE_APPLICATION_CONTEXTS=1 -DCAPSTONE_APPLICATION_GRANT_BYTES=83886080" \
    "CPYD_HEAP=sublet-pymalloc"
echo
echo "=== it must refuse a contradiction, not silently pick one ==="
chk "7. CPYD_HEAP=sublet + CPY_SUBLET=1 -> refused" \
    "EXIT=2" "CPYD_HEAP=sublet; CPY_SUBLET=1"
chk "8. CPYD_HEAP=typo                  -> refused" \
    "EXIT=2" "CPYD_HEAP=subletpymalloc"
echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
