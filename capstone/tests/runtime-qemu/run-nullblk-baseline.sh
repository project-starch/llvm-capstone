#!/usr/bin/env bash
set -euo pipefail

# Regression wrapper for the in-kernel reference null_blk path.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-nullb-baseline.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --buildroot-dir "$CAPSTONE_BUILDROOT_DIR" \
  --qemu-binary "$CAPSTONE_QEMU_BINARY" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "dmesg -n 8 && modprobe configfs && cd /nullb/baseline && insmod ./null_blk.ko && test -b /dev/nullb0 && echo hello-world | dd of=/dev/nullb0 bs=1024 count=1 && dd if=/dev/nullb0 bs=1024 count=1 | hexdump -C && rmmod null_blk" \
  --success-marker "null_blk: disk nullb0 created" \
  --success-marker "1+0 records in" \
  --success-marker "1+0 records out"

echo "run-nullblk-baseline.sh wrapper completed. Full serial log: $LOG_FILE"

