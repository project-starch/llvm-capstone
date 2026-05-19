#!/usr/bin/env bash
set -euo pipefail

# One-command wrapper for the first SQLite-facing path-access proof.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-hostcall-path-access-probe.log}
EXISTING_PATH=/tmp/hostcall_v0_path_access_present.txt
MISSING_PATH=/tmp/hostcall_v0_path_access_missing.txt
EXPECTED_TEXT='hostcall-v0 path access payload'

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/hostcall_path_access_probe.user \
      "$SHARE_DIR"/hostcall_path_access_probe.smode

bash "$SCRIPT_DIR/build-hostcall-path-access-probe.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "rm -f $MISSING_PATH && printf '%s' '$EXPECTED_TEXT' > $EXISTING_PATH && cp /mnt/host/hostcall_path_access_probe.user /tmp/hostcall_path_access_probe.user && chmod 0755 /tmp/hostcall_path_access_probe.user && /tmp/hostcall_path_access_probe.user /mnt/host/hostcall_path_access_probe.smode && echo __HOSTCALL_PATH_ACCESS_OK__" \
  --success-marker "hostcall-path-access-probe: first call retval = 1" \
  --success-marker "hostcall-path-access-probe: servicing HC_V0_OP_PATH_ACCESS for /tmp/hostcall_v0_path_access_present.txt" \
  --success-marker "hostcall-path-access-probe: second call retval = 1" \
  --success-marker "hostcall-path-access-probe: servicing HC_V0_OP_PATH_ACCESS for /tmp/hostcall_v0_path_access_missing.txt" \
  --success-marker "hostcall-path-access-probe: third call retval = 0" \
  --success-marker "hostcall-path-access-probe: success" \
  --success-marker "__HOSTCALL_PATH_ACCESS_OK__"

echo "run-hostcall-path-access-probe.sh wrapper completed. Full serial log: $LOG_FILE"

