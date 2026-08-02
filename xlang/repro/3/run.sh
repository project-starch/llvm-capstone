#!/usr/bin/env bash
# Run Row 3 (GHSA-f56g-chqp-22m9) and show the violation.
# PASS = valgrind reports an invalid read in pa_proplist_iterate, with the free
#        attributed to Proplist::drop reached from into_iter.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$DIR/target/x86_64-unknown-linux-gnu/debug/libpulse-repro"

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found -- run ./build.sh first" >&2
    exit 1
fi

SYMBOLIZER="$(command -v llvm-symbolizer || echo /usr/lib/llvm-21/bin/llvm-symbolizer)"

echo "=== 1/2: native + ASan (expected to complete WITHOUT reporting) ==="
# This is not a failure. The stale dereference happens inside libpulse.so, a
# distro shared library with no sanitizer instrumentation, so ASan has nothing to
# check even though it poisoned the region on free. Shown explicitly so the clean
# result is not mistaken for "the bug does not reproduce".
ASAN_SYMBOLIZER_PATH="$SYMBOLIZER" "$BIN" || true
echo ""

echo "=== 2/2: valgrind memcheck (expect 'Invalid read' in pa_proplist_iterate) ==="
# Valgrind instruments at the machine level and catches accesses made by
# uninstrumented libraries, so it is the ground-truth tool for this row.
if ! command -v valgrind >/dev/null; then
    echo "error: valgrind not installed -- it is required for this row." >&2
    exit 1
fi
valgrind --num-callers=8 --error-exitcode=9 "$BIN" 2>&1 \
  | grep -E 'Invalid read|pa_proplist|pa_hashmap|proplist\.rs|main\.rs|free.d|ERROR SUMMARY'
VG_EXIT=${PIPESTATUS[0]}
echo ""
echo "valgrind exited with: $VG_EXIT (9 = errors found, as expected)"

echo ""
echo "=== RISC-V QEMU ==="
echo "Not built for this row -- see README.md 'RISC-V QEMU'. Task spec §9 accepts"
echo "the native reproduction plus boundary.md as a complete artifact."

if [ "$VG_EXIT" -eq 9 ]; then
    echo ""
    echo "PASS: valgrind reported the use-after-free."
else
    echo ""
    echo "FAIL: expected valgrind exit 9 (errors found), got $VG_EXIT." >&2
    exit 1
fi
