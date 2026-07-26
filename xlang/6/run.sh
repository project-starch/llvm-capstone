#!/usr/bin/env bash
# Run Row 6 (CVE-2026-1979 / mruby #6701) and show the violation.
# PASS = native ASan reports heap-buffer-overflow WRITE at vm.c:1788.
# NOTE: heap-buffer-OVERFLOW, not use-after-free -- see target.md.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIGGER="$DIR/trigger.rb"

echo "=== Running native + ASan (expect heap-buffer-overflow WRITE abort) ==="
"$DIR/mruby/build/host/bin/mruby" "$TRIGGER"
NATIVE_EXIT=$?
echo "Native run exited with: $NATIVE_EXIT"

echo ""
echo "=== Running under RISC-V QEMU (expect SIGSEGV) ==="
# Observed: segmentation fault, exit 139, on 3/3 runs. The corrupted LOADI_5 R38
# stores past the end of the VM stack allocation; on the riscv64 build that
# reaches an unmapped page. "done" is printed first -- the errant write happens
# during the recursion and the fault surfaces slightly later.
qemu-riscv64 -L /usr/riscv64-linux-gnu "$DIR/mruby/build/riscv64/bin/mruby" "$TRIGGER"
QEMU_EXIT=$?
echo "RISC-V QEMU run exited with: $QEMU_EXIT (expected 139 = SIGSEGV)"

echo ""
echo "=== Corrupted instruction, straight from the compiler ==="
# LOADI_5 should target R2; the peephole bug rewrites the operand to R38 (=OP_JMPIF).
"$DIR/mruby/build/host/bin/mrbc" -v -o /dev/null "$TRIGGER" 2>&1 | grep -A3 'ENTER' | head -4

if [ "$NATIVE_EXIT" -eq 1 ]; then
    echo ""
    echo "PASS: ASan aborted as expected."
else
    echo ""
    echo "FAIL: expected an ASan abort (exit 1), got exit $NATIVE_EXIT." >&2
    exit 1
fi
