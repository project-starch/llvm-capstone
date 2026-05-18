#!/usr/bin/env bash
set -euo pipefail

# One-command wrapper for the first handle-based FILE_TRUNCATE proof.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-hostcall-file-handle-truncate-probe.log}
INPUT_PATH=/tmp/hostcall_v0_handle_truncate.txt
INITIAL_TEXT='hostcall-v0 handle truncate payload'
TARGET_SIZE=9

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/hostcall_file_handle_truncate_probe.user \
      "$SHARE_DIR"/hostcall_file_handle_truncate_probe.smode

bash "$SCRIPT_DIR/build-hostcall-file-handle-truncate-probe.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "printf '%s' '$INITIAL_TEXT' > $INPUT_PATH && cp /mnt/host/hostcall_file_handle_truncate_probe.user /tmp/hostcall_file_handle_truncate_probe.user && chmod 0755 /tmp/hostcall_file_handle_truncate_probe.user && /tmp/hostcall_file_handle_truncate_probe.user /mnt/host/hostcall_file_handle_truncate_probe.smode && test \"\$(wc -c < $INPUT_PATH)\" = \"$TARGET_SIZE\" && echo __HOSTCALL_FILE_HANDLE_TRUNCATE_OK__" \
  --success-marker "hostcall-file-handle-truncate-probe: first call retval = 1" \
  --success-marker "hostcall-file-handle-truncate-probe: servicing HC_V0_OP_FILE_OPEN" \
  --success-marker "hostcall-file-handle-truncate-probe: second call retval = 1" \
  --success-marker "hostcall-file-handle-truncate-probe: servicing HC_V0_OP_FILE_TRUNCATE" \
  --success-marker "hostcall-file-handle-truncate-probe: third call retval = 1" \
  --success-marker "hostcall-file-handle-truncate-probe: servicing HC_V0_OP_FILE_STAT_BASIC" \
  --success-marker "hostcall-file-handle-truncate-probe: fourth call retval = 1" \
  --success-marker "hostcall-file-handle-truncate-probe: servicing HC_V0_OP_FILE_CLOSE" \
  --success-marker "hostcall-file-handle-truncate-probe: fifth call retval = 0" \
  --success-marker "hostcall-file-handle-truncate-probe: success" \
  --success-marker "__HOSTCALL_FILE_HANDLE_TRUNCATE_OK__"

echo "run-hostcall-file-handle-truncate-probe.sh wrapper completed. Full serial log: $LOG_FILE"

