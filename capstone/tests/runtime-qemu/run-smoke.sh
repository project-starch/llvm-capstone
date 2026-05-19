#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-smoke.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/*.dom "$SHARE_DIR"/capstone-test.user

bash "$SCRIPT_DIR/build-capstone-test-user.sh" \
  "$SHARE_DIR/capstone-test.user"

bash "$SCRIPT_DIR/build-domain.sh" \
  "$SCRIPT_DIR/domains/write_42.c" \
  "$SHARE_DIR/write_42.dom"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command '/mnt/host/capstone-test.user /mnt/host/write_42.dom' \
  --success-marker 'Created domain ID = 0' \
  --success-marker 'Called dom (1-th time) retval = 42'

echo "run-smoke.sh wrapper completed. Full serial log: $LOG_FILE"

