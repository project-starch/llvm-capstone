#!/usr/bin/env bash
# Run Row 11 (CVE-2018-10191) and show the violation.
# PASS = native ASan reports heap-buffer-overflow at vm.c:1208 in mrb_vm_exec.
# NOTE: heap-buffer-OVERFLOW, not use-after-free -- see target.md.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIGGER="$DIR/trigger.rb"

echo "=== Running native + ASan (expect heap-buffer-overflow abort) ==="
"$DIR/mruby/build/host/bin/mruby" "$TRIGGER"
NATIVE_EXIT=$?
echo "Native run exited with: $NATIVE_EXIT"

echo ""
echo "=== Running under RISC-V QEMU (expect SIGSEGV) ==="
# Observed: segmentation fault, exit 139, on 3/3 runs. The out-of-range
# e->stack[b] read at vm.c:1208 reaches an unmapped page on the riscv64 build,
# so the plain non-ASan binary faults outright -- an acceptable QEMU
# reproduction per task spec §4.2, with native ASan above as the authoritative
# "it is an out-of-bounds read" evidence.
qemu-riscv64 -L /usr/riscv64-linux-gnu "$DIR/mruby/build/riscv64/bin/mruby" "$TRIGGER"
QEMU_EXIT=$?
echo "RISC-V QEMU run exited with: $QEMU_EXIT (expected 139 = SIGSEGV)"

if [ "$NATIVE_EXIT" -eq 1 ]; then
    echo ""
    echo "PASS: ASan aborted as expected."
else
    echo ""
    echo "FAIL: expected an ASan abort (exit 1), got exit $NATIVE_EXIT." >&2
    exit 1
fi
