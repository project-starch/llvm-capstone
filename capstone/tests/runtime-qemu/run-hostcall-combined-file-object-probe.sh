#!/usr/bin/env bash
set -euo pipefail

# One-command wrapper for the first combined file-object proof:
# OPEN -> WRITE -> SYNC -> CLOSE -> OPEN -> READ -> CLOSE.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-hostcall-combined-file-object-probe.log}
OUTPUT_PATH=/tmp/hostcall_v0_combined_file_object.txt
EXPECTED_TEXT='hostcall-v0 combined file object payload'

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/hostcall_combined_file_object_probe.user \
      "$SHARE_DIR"/hostcall_combined_file_object_probe.smode

bash "$SCRIPT_DIR/build-hostcall-combined-file-object-probe.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "rm -f $OUTPUT_PATH && cp /mnt/host/hostcall_combined_file_object_probe.user /tmp/hostcall_combined_file_object_probe.user && chmod 0755 /tmp/hostcall_combined_file_object_probe.user && /tmp/hostcall_combined_file_object_probe.user /mnt/host/hostcall_combined_file_object_probe.smode && test \"\$(cat $OUTPUT_PATH)\" = \"$EXPECTED_TEXT\" && echo __HOSTCALL_COMBINED_FILE_OBJECT_OK__" \
  --success-marker "hostcall-combined-file-object-probe: first call retval = 1" \
  --success-marker "hostcall-combined-file-object-probe: servicing HC_V0_OP_FILE_WRITE" \
  --success-marker "hostcall-combined-file-object-probe: servicing HC_V0_OP_FILE_SYNC" \
  --success-marker "hostcall-combined-file-object-probe: servicing HC_V0_OP_FILE_READ" \
  --success-marker "hostcall-combined-file-object-probe: payload revoked and re-shared for read response plus final close request" \
  --success-marker "hostcall-combined-file-object-probe: eighth call retval = 0" \
  --success-marker "hostcall-combined-file-object-probe: success" \
  --success-marker "__HOSTCALL_COMBINED_FILE_OBJECT_OK__"

echo "run-hostcall-combined-file-object-probe.sh wrapper completed. Full serial log: $LOG_FILE"

