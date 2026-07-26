#!/usr/bin/env bash
# Run Row 7's probe. This row is NOT REPRODUCED -- read target.md first.
# The probe drives the path the spec describes and is EXPECTED to complete
# cleanly; that is the negative result, not a broken trigger.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIGGER="$DIR/trigger.rb"

echo "=== Row 7: NOT REPRODUCED (see target.md) ==="
echo "Expected outcome below is a clean completion, not a crash."
echo ""

echo "=== Running native + ASan ==="
"$DIR/mruby/build/host/bin/mruby" "$TRIGGER"
NATIVE_EXIT=$?
echo "Native run exited with: $NATIVE_EXIT (expected 0)"

echo ""
echo "=== Running under RISC-V QEMU ==="
qemu-riscv64 -L /usr/riscv64-linux-gnu "$DIR/mruby/build/riscv64/bin/mruby" "$TRIGGER"
QEMU_EXIT=$?
echo "RISC-V QEMU run exited with: $QEMU_EXIT (expected 0)"

echo ""
if [ "$NATIVE_EXIT" -eq 0 ] && [ "$QEMU_EXIT" -eq 0 ]; then
    echo "As documented: the described GC hazard in mrb_bint_reduce is closed by"
    echo "mruby's allocation arena. See target.md for the evidence and for what"
    echo "would be needed to settle this row."
else
    echo "UNEXPECTED: a fault occurred. If ASan reported a use-after-free, this row"
    echo "may be reproducible after all -- capture the trace and revisit target.md." >&2
fi
