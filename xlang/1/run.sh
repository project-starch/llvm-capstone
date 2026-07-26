#!/usr/bin/env bash
# Run Row 1 (rlua #19) and show the violation.
# PASS = AddressSanitizer reports heap-use-after-free with the stale read in
#        `Userdata::add_methods::{closure#0}` (src/main.rs) reached from Lua.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$DIR/target/x86_64-unknown-linux-gnu/debug/rlua-repro"

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found -- run ./build.sh first" >&2
    exit 1
fi

# ASan cannot symbolize Rust frames without an explicit symbolizer on most
# distros; without this the trace is bare hex addresses and useless as evidence.
SYMBOLIZER="$(command -v llvm-symbolizer || echo /usr/lib/llvm-21/bin/llvm-symbolizer)"

echo "=== Running native + ASan (expect heap-use-after-free abort) ==="
ASAN_SYMBOLIZER_PATH="$SYMBOLIZER" "$BIN"
EXIT_CODE=$?
echo "Native run exited with: $EXIT_CODE"

echo ""
echo "=== RISC-V QEMU ==="
echo "Not built for this row -- see README.md 'RISC-V QEMU'. Per task spec §9 the"
echo "native ASan reproduction plus boundary.md is a complete artifact when the"
echo "QEMU leg is impractical."

# Expected: EXIT_CODE=1 (ASan abort). Anything else is a failed reproduction.
if [ "$EXIT_CODE" -eq 1 ]; then
    echo ""
    echo "PASS: ASan aborted as expected."
else
    echo ""
    echo "FAIL: expected an ASan abort (exit 1), got exit $EXIT_CODE." >&2
    exit 1
fi
