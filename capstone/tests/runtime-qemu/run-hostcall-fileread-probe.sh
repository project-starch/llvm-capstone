#!/usr/bin/env bash
set -euo pipefail

# One-command regression wrapper for the first reverse-direction HostCall-style
# proof: the domain requests a read-like service and the helper returns borrowed
# response bytes through the payload region on round 2.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-hostcall-fileread-probe.log}
INPUT_PATH=/tmp/hostcall_v0_read_source.txt
EXPECTED_TEXT='hostcall-v0 input payload'

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/hostcall_fileread_probe.user "$SHARE_DIR"/hostcall_fileread_probe.smode

# Rebuild the reverse-direction probe artifacts into the 9p-shared directory used by QEMU.
bash "$SCRIPT_DIR/build-hostcall-fileread-probe.sh" "$SHARE_DIR"

# The wrapper succeeds only if the helper reads the fixed guest-side file, shares the
# response payload back into the domain, and the domain validates that content.
python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "printf '%s' '$EXPECTED_TEXT' > $INPUT_PATH && cp /mnt/host/hostcall_fileread_probe.user /tmp/hostcall_fileread_probe.user && chmod 0755 /tmp/hostcall_fileread_probe.user && /tmp/hostcall_fileread_probe.user /mnt/host/hostcall_fileread_probe.smode && echo __HOSTCALL_FILEREAD_OK__" \
  --success-marker "hostcall-fileread-probe: first call retval = 1" \
  --success-marker "hostcall-fileread-probe: servicing HC_V0_OP_READ_GUEST_TMPFILE" \
  --success-marker "hostcall-fileread-probe: payload shared as borrowed-in response" \
  --success-marker "hostcall-fileread-probe: second call retval = 0" \
  --success-marker "hostcall-fileread-probe: success" \
  --success-marker "__HOSTCALL_FILEREAD_OK__"

echo "run-hostcall-fileread-probe.sh wrapper completed. Full serial log: $LOG_FILE"

