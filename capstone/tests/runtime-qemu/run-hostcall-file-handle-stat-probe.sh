#!/usr/bin/env bash
set -euo pipefail

# One-command wrapper for the first handle-based FILE_STAT_BASIC proof.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-hostcall-file-handle-stat-probe.log}
INPUT_PATH=/tmp/hostcall_v0_handle_stat.txt
EXPECTED_TEXT='hostcall-v0 handle stat payload'

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/hostcall_file_handle_stat_probe.user \
      "$SHARE_DIR"/hostcall_file_handle_stat_probe.smode

bash "$SCRIPT_DIR/build-hostcall-file-handle-stat-probe.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "printf '%s' '$EXPECTED_TEXT' > $INPUT_PATH && cp /mnt/host/hostcall_file_handle_stat_probe.user /tmp/hostcall_file_handle_stat_probe.user && chmod 0755 /tmp/hostcall_file_handle_stat_probe.user && /tmp/hostcall_file_handle_stat_probe.user /mnt/host/hostcall_file_handle_stat_probe.smode && echo __HOSTCALL_FILE_HANDLE_STAT_OK__" \
  --success-marker "hostcall-file-handle-stat-probe: first call retval = 1" \
  --success-marker "hostcall-file-handle-stat-probe: servicing HC_V0_OP_FILE_OPEN" \
  --success-marker "hostcall-file-handle-stat-probe: second call retval = 1" \
  --success-marker "hostcall-file-handle-stat-probe: servicing HC_V0_OP_FILE_STAT_BASIC" \
  --success-marker "hostcall-file-handle-stat-probe: third call retval = 1" \
  --success-marker "hostcall-file-handle-stat-probe: servicing HC_V0_OP_FILE_CLOSE" \
  --success-marker "hostcall-file-handle-stat-probe: fourth call retval = 0" \
  --success-marker "hostcall-file-handle-stat-probe: success" \
  --success-marker "__HOSTCALL_FILE_HANDLE_STAT_OK__"

echo "run-hostcall-file-handle-stat-probe.sh wrapper completed. Full serial log: $LOG_FILE"

