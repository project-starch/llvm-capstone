#!/usr/bin/env bash
set -euo pipefail

# One-command wrapper for the first handle-based FILE_SYNC proof.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-hostcall-file-handle-sync-probe.log}
OUTPUT_PATH=/tmp/hostcall_v0_handle_sync.txt
EXPECTED_TEXT='hostcall-v0 handle sync payload'

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/hostcall_file_handle_sync_probe.user \
      "$SHARE_DIR"/hostcall_file_handle_sync_probe.smode

bash "$SCRIPT_DIR/build-hostcall-file-handle-sync-probe.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "cp /mnt/host/hostcall_file_handle_sync_probe.user /tmp/hostcall_file_handle_sync_probe.user && chmod 0755 /tmp/hostcall_file_handle_sync_probe.user && rm -f $OUTPUT_PATH && /tmp/hostcall_file_handle_sync_probe.user /mnt/host/hostcall_file_handle_sync_probe.smode && test \"\$(cat $OUTPUT_PATH)\" = \"$EXPECTED_TEXT\" && echo __HOSTCALL_FILE_HANDLE_SYNC_OK__" \
  --success-marker "hostcall-file-handle-sync-probe: first call retval = 1" \
  --success-marker "hostcall-file-handle-sync-probe: servicing HC_V0_OP_FILE_OPEN" \
  --success-marker "hostcall-file-handle-sync-probe: second call retval = 1" \
  --success-marker "hostcall-file-handle-sync-probe: servicing HC_V0_OP_FILE_WRITE" \
  --success-marker "hostcall-file-handle-sync-probe: third call retval = 1" \
  --success-marker "hostcall-file-handle-sync-probe: servicing HC_V0_OP_FILE_SYNC" \
  --success-marker "hostcall-file-handle-sync-probe: fourth call retval = 1" \
  --success-marker "hostcall-file-handle-sync-probe: servicing HC_V0_OP_FILE_CLOSE" \
  --success-marker "hostcall-file-handle-sync-probe: fifth call retval = 0" \
  --success-marker "hostcall-file-handle-sync-probe: success" \
  --success-marker "__HOSTCALL_FILE_HANDLE_SYNC_OK__"

echo "run-hostcall-file-handle-sync-probe.sh wrapper completed. Full serial log: $LOG_FILE"

