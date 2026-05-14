#!/usr/bin/env bash
set -euo pipefail

# One-command regression wrapper for the second HostCall-style proof: the domain
# still produces a borrowed payload, but the helper now uses ordinary guest Linux
# file I/O instead of stdout.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-hostcall-filewrite-probe.log}
OUTPUT_PATH=/tmp/hostcall_v0_filewrite.txt
EXPECTED_TEXT='hostcall-v0 file payload'

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/hostcall_filewrite_probe.user "$SHARE_DIR"/hostcall_filewrite_probe.smode

# Rebuild the second probe artifacts into the 9p-shared directory used by QEMU.
bash "$SCRIPT_DIR/build-hostcall-filewrite-probe.sh" "$SHARE_DIR"

# The wrapper succeeds only if the two-round protocol completes and the helper-side
# file write leaves the exact expected content in the guest tmp file.
python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "cp /mnt/host/hostcall_filewrite_probe.user /tmp/hostcall_filewrite_probe.user && chmod 0755 /tmp/hostcall_filewrite_probe.user && rm -f $OUTPUT_PATH && /tmp/hostcall_filewrite_probe.user /mnt/host/hostcall_filewrite_probe.smode && test \"\$(cat $OUTPUT_PATH)\" = \"$EXPECTED_TEXT\" && echo __HOSTCALL_FILEWRITE_OK__" \
  --success-marker "hostcall-filewrite-probe: first call retval = 1" \
  --success-marker "hostcall-filewrite-probe: servicing HC_V0_OP_WRITE_GUEST_TMPFILE" \
  --success-marker "hostcall-filewrite-probe: second call retval = 0" \
  --success-marker "hostcall-filewrite-probe: success" \
  --success-marker "__HOSTCALL_FILEWRITE_OK__"

echo "run-hostcall-filewrite-probe.sh wrapper completed. Full serial log: $LOG_FILE"

