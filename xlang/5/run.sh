#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIGGER="$DIR/trigger.rb"

echo "=== Running Native ASan (Expect heap-use-after-free abort) ==="
set +e
"$DIR/mruby/build/host-asan/bin/mruby" "$TRIGGER"
EXIT_CODE=$?
set -e
echo "Native run exited with: $EXIT_CODE"

echo ""
echo "=== Running under RISC-V QEMU (Expect run or behavior) ==="
set +e
qemu-riscv64 -L /usr/riscv64-linux-gnu "$DIR/mruby/build/riscv64/bin/mruby" "$TRIGGER"
QEMU_EXIT_CODE=$?
set -e
echo "RISC-V QEMU run exited with: $QEMU_EXIT_CODE"
