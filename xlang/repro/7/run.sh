#!/usr/bin/env bash
# Run Row 7 (RUSTSEC-2022-0070) and show the violation.
#
# PASS = valgrind reports an invalid read inside a freed 208-byte block -- the
#        preallocated secp256k1 context, freed when the Vec backing it went out
#        of scope while the context escaped on a bogus 'static lifetime.
#
# The plain run is shown FIRST and completes cleanly on purpose. Without an
# instrument the stale read lands on memory the allocator has not reused, so a
# valid-looking signature comes back. That clean result is evidence about
# detectability, not evidence that the defect is absent.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$DIR/target/debug/secp256k1-repro"

if [ ! -x "$BIN" ]; then
  echo "error: $BIN not found -- run ./build.sh first" >&2
  exit 1
fi

echo "=== 1/2: plain run (expected to COMPLETE, printing a signature) ==="
"$BIN" || true
echo ""
echo "=== 2/2: valgrind memcheck (expect 'Invalid read' in a free'd block) ==="
valgrind --error-exitcode=9 --num-callers=8 "$BIN"
rc=$?
echo ""
if [ "$rc" -eq 9 ]; then
  echo "PASS: valgrind reported the use-after-free (exit 9)"
  exit 0
fi
echo "FAIL: expected valgrind exit 9, got $rc" >&2
exit 1
