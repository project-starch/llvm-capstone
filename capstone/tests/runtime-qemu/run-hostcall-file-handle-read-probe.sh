#!/usr/bin/env bash
set -euo pipefail

# One-command wrapper for the first handle-based FILE_READ proof.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-hostcall-file-handle-read-probe.log}
INPUT_PATH=/tmp/hostcall_v0_handle_read.txt
EXPECTED_TEXT='hostcall-v0 handle read payload'

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/hostcall_file_handle_read_probe.user \
      "$SHARE_DIR"/hostcall_file_handle_read_probe.smode

bash "$SCRIPT_DIR/build-hostcall-file-handle-read-probe.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "printf '%s' '$EXPECTED_TEXT' > $INPUT_PATH && cp /mnt/host/hostcall_file_handle_read_probe.user /tmp/hostcall_file_handle_read_probe.user && chmod 0755 /tmp/hostcall_file_handle_read_probe.user && /tmp/hostcall_file_handle_read_probe.user /mnt/host/hostcall_file_handle_read_probe.smode && echo __HOSTCALL_FILE_HANDLE_READ_OK__" \
  --success-marker "hostcall-file-handle-read-probe: first call retval = 1" \
  --success-marker "hostcall-file-handle-read-probe: servicing HC_V0_OP_FILE_OPEN" \
  --success-marker "hostcall-file-handle-read-probe: second call retval = 1" \
  --success-marker "hostcall-file-handle-read-probe: servicing HC_V0_OP_FILE_READ" \
  --success-marker "hostcall-file-handle-read-probe: payload revoked and re-shared for read response" \
  --success-marker "hostcall-file-handle-read-probe: third call retval = 0" \
  --success-marker "hostcall-file-handle-read-probe: success" \
  --success-marker "__HOSTCALL_FILE_HANDLE_READ_OK__"

echo "run-hostcall-file-handle-read-probe.sh wrapper completed. Full serial log: $LOG_FILE"

