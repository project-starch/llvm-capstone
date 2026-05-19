#!/usr/bin/env bash
set -euo pipefail

# One-command wrapper for the first SQLite-facing path-delete proof.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-hostcall-path-delete-probe.log}
DELETE_PATH=/tmp/hostcall_v0_path_delete_target.txt
EXPECTED_TEXT='hostcall-v0 path delete payload'

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/hostcall_path_delete_probe.user \
      "$SHARE_DIR"/hostcall_path_delete_probe.smode

bash "$SCRIPT_DIR/build-hostcall-path-delete-probe.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "printf '%s' '$EXPECTED_TEXT' > $DELETE_PATH && cp /mnt/host/hostcall_path_delete_probe.user /tmp/hostcall_path_delete_probe.user && chmod 0755 /tmp/hostcall_path_delete_probe.user && /tmp/hostcall_path_delete_probe.user /mnt/host/hostcall_path_delete_probe.smode && test ! -e $DELETE_PATH && echo __HOSTCALL_PATH_DELETE_OK__" \
  --success-marker "hostcall-path-delete-probe: first call retval = 1" \
  --success-marker "hostcall-path-delete-probe: servicing HC_V0_OP_PATH_DELETE for /tmp/hostcall_v0_path_delete_target.txt" \
  --success-marker "hostcall-path-delete-probe: second call retval = 1" \
  --success-marker "hostcall-path-delete-probe: servicing HC_V0_OP_PATH_ACCESS for /tmp/hostcall_v0_path_delete_target.txt" \
  --success-marker "hostcall-path-delete-probe: third call retval = 0" \
  --success-marker "hostcall-path-delete-probe: success" \
  --success-marker "__HOSTCALL_PATH_DELETE_OK__"

echo "run-hostcall-path-delete-probe.sh wrapper completed. Full serial log: $LOG_FILE"

