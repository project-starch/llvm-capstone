#!/usr/bin/env bash
set -euo pipefail

# Regression wrapper for the split null_blk I/O path through the restored runtime.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-nullb-split-io.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --buildroot-dir "$CAPSTONE_BUILDROOT_DIR" \
  --qemu-binary "$CAPSTONE_QEMU_BINARY" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "dmesg -n 1 && modprobe configfs && /null_blk.user && insmod /nullb/capstone_split/null_blk.ko && test -b /dev/nullb0 && echo __SPLIT_READY__ && echo hello-world | dd of=/dev/nullb0 bs=1024 count=1 && dd if=/dev/nullb0 bs=1024 count=1 | hexdump -C && echo __SPLIT_DONE__" \
  --success-marker "SBI domain created with ID 0" \
  --success-marker "__SPLIT_READY__" \
  --success-marker "__SPLIT_DONE__"

echo "run-nullblk-split-io.sh wrapper completed. Full serial log: $LOG_FILE"
