#!/usr/bin/env bash
set -euo pipefail

# One-command wrapper for the first helper-managed file-object proof:
# open an existing file through the helper, hand back a protocol token, then
# close that token on the next round using the same borrowed payload region.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-hostcall-file-open-close-probe.log}
INPUT_PATH=/tmp/hostcall_v0_open_close_source.txt
EXPECTED_TEXT='hostcall-v0 open-close source'

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/hostcall_file_open_close_probe.user \
      "$SHARE_DIR"/hostcall_file_open_close_probe.smode

bash "$SCRIPT_DIR/build-hostcall-file-open-close-probe.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "printf '%s' '$EXPECTED_TEXT' > $INPUT_PATH && cp /mnt/host/hostcall_file_open_close_probe.user /tmp/hostcall_file_open_close_probe.user && chmod 0755 /tmp/hostcall_file_open_close_probe.user && /tmp/hostcall_file_open_close_probe.user /mnt/host/hostcall_file_open_close_probe.smode && test \"\$(cat $INPUT_PATH)\" = \"$EXPECTED_TEXT\" && echo __HOSTCALL_FILE_OPEN_CLOSE_OK__" \
  --success-marker "hostcall-file-open-close-probe: first call retval = 1" \
  --success-marker "hostcall-file-open-close-probe: servicing HC_V0_OP_FILE_OPEN" \
  --success-marker "hostcall-file-open-close-probe: payload revoked and re-shared for close request" \
  --success-marker "hostcall-file-open-close-probe: servicing HC_V0_OP_FILE_CLOSE" \
  --success-marker "hostcall-file-open-close-probe: third call retval = 0" \
  --success-marker "hostcall-file-open-close-probe: success" \
  --success-marker "__HOSTCALL_FILE_OPEN_CLOSE_OK__"

echo "run-hostcall-file-open-close-probe.sh wrapper completed. Full serial log: $LOG_FILE"

