#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-nullb-split-rmmod.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --buildroot-dir "$CAPSTONE_BUILDROOT_DIR" \
  --qemu-binary "$CAPSTONE_QEMU_BINARY" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "dmesg -n 8 && modprobe configfs && /null_blk.user && insmod /nullb/capstone_split/null_blk.ko && echo __BEFORE_RMMOD__ && rmmod null_blk && echo __AFTER_RMMOD__" \
  --success-marker "SBI domain created with ID 0" \
  --success-marker "__BEFORE_RMMOD__" \
  --success-marker "__AFTER_RMMOD__"

echo "run-nullblk-split-rmmod.sh wrapper completed. Full serial log: $LOG_FILE"

