#!/usr/bin/env bash
# Run Row 2 (rlua #97) and show the violation.
# PASS = AddressSanitizer reports a temporal violation with the stale read in
#        <rlua::table::Table>::len -- the escaped handle used after drop(lua).
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$DIR/target/x86_64-unknown-linux-gnu/release/rlua-repro-2"

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found -- run ./build.sh first" >&2
    exit 1
fi

SYMBOLIZER="$(command -v llvm-symbolizer || echo /usr/lib/llvm-21/bin/llvm-symbolizer)"

echo "=== Half the bug: this must NOT compile against a fixed rlua ==="
echo "src/main.rs is upstream's compile-fail test (tests/compile-fail/"
echo "static_callback_args_tls.rs, added by fix commit 0f5a9a3). That it compiles"
echo "at all against the pinned commit is the unsoundness."
echo ""

echo "=== Running native + ASan (expect a use-after-lifetime abort) ==="
ASAN_SYMBOLIZER_PATH="$SYMBOLIZER" "$BIN"
EXIT_CODE=$?
echo "Native run exited with: $EXIT_CODE"

echo ""
echo "=== RISC-V QEMU ==="
echo "Not built for this row -- see README.md 'RISC-V QEMU'. Task spec §9 accepts"
echo "the native ASan reproduction plus boundary.md as a complete artifact."

if [ "$EXIT_CODE" -eq 1 ]; then
    echo ""
    echo "PASS: ASan aborted as expected."
else
    echo ""
    echo "FAIL: expected an ASan abort (exit 1), got exit $EXIT_CODE." >&2
    exit 1
fi
