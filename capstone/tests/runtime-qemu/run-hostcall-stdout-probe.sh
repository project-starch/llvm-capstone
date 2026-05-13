#!/usr/bin/env bash
set -euo pipefail

# One-command regression wrapper for the first validated HostCall-style
# WRITE_STDOUT proof on top of the restored shared-region runtime baseline.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-hostcall-stdout-probe.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/hostcall_stdout_probe.user "$SHARE_DIR"/hostcall_stdout_probe.smode

# Rebuild the probe artifacts into the host-shared 9p directory used by QEMU.
bash "$SCRIPT_DIR/build-hostcall-stdout-probe.sh" "$SHARE_DIR"

# The wrapper succeeds only if the guest reaches the expected two-round protocol
# markers and the payload is actually printed once from the host helper.
python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "cp /mnt/host/hostcall_stdout_probe.user /tmp/hostcall_stdout_probe.user && chmod 0755 /tmp/hostcall_stdout_probe.user && /tmp/hostcall_stdout_probe.user /mnt/host/hostcall_stdout_probe.smode" \
  --success-marker "hostcall-stdout-probe: first call retval = 1" \
  --success-marker "hostcall-v0 payload from domain" \
  --success-marker "hostcall-stdout-probe: second call retval = 0" \
  --success-marker "hostcall-stdout-probe: success"

echo "run-hostcall-stdout-probe.sh wrapper completed. Full serial log: $LOG_FILE"


