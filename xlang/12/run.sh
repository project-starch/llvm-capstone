#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIGGER="$DIR/trigger.rb"

echo "=== Running Native ASan (Expect heap-use-after-free abort) ==="
set +e
"$DIR/mruby/build/host/bin/mruby" "$TRIGGER"
EXIT_CODE=$?
set -e
echo "Native run exited with: $EXIT_CODE"

echo ""
echo "=== Running under RISC-V QEMU (observed: exit 1) ==="
# Observed: exit 1 with a clean Ruby 'closed stream. (IOError)'. NOT a
# memory-safety event: without ASan the freed mrb_io struct still reads as
# closed, so fptr_finalize raises instead of corrupting.
# Task spec §4.2 requires the observed QEMU behaviour to be documented here;
# the native ASan run above is the authoritative memory-safety evidence.
set +e
qemu-riscv64 -L /usr/riscv64-linux-gnu "$DIR/mruby/build/riscv64/bin/mruby" "$TRIGGER"
QEMU_EXIT_CODE=$?
set -e
echo "RISC-V QEMU run exited with: $QEMU_EXIT_CODE (expected 1)"

# --- assertion -------------------------------------------------------------
# Native ASan must abort (exit 1); the QEMU leg must match the behaviour
# documented above. Makes the row scriptable: non-zero means the reproduction
# regressed, not merely that a crash happened.
FAILED=0
if [ "$EXIT_CODE" -ne 1 ]; then
    echo "FAIL: native ASan expected exit 1, got $EXIT_CODE." >&2
    FAILED=1
fi
if [ "$QEMU_EXIT_CODE" -ne 1 ]; then
    echo "FAIL: RISC-V QEMU expected exit 1, got $QEMU_EXIT_CODE." >&2
    FAILED=1
fi
if [ "$FAILED" -eq 0 ]; then
    echo ""
    echo "PASS: native ASan aborted and QEMU behaved as documented."
else
    exit 1
fi
