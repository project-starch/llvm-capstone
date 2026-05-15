#!/usr/bin/env bash
set -euo pipefail

# One-command wrapper for the workaround variant: explicitly revoke the payload
# region before re-sharing it for the second borrowed-output round.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-hostcall-second-pending-payload-revoke-probe.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/hostcall_second_pending_payload_revoke_probe.user \
      "$SHARE_DIR"/hostcall_second_pending_payload_revoke_probe.smode

bash "$SCRIPT_DIR/build-hostcall-second-pending-payload-revoke-probe.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "cp /mnt/host/hostcall_second_pending_payload_revoke_probe.user /tmp/hostcall_second_pending_payload_revoke_probe.user && chmod 0755 /tmp/hostcall_second_pending_payload_revoke_probe.user && /tmp/hostcall_second_pending_payload_revoke_probe.user /mnt/host/hostcall_second_pending_payload_revoke_probe.smode" \
  --success-marker "hostcall-second-pending-payload-probe: revoking payload region before round 2 re-share" \
  --success-marker "hostcall-second-pending-payload-probe: payload re-shared as borrowed-out for round 2" \
  --success-marker "hostcall-second-pending-payload-probe: second pending with payload reuse observed" \
  --success-marker "hostcall-second-pending-payload-probe: success" \
  --success-marker "__HOSTCALL_SECOND_PENDING_PAYLOAD_OK__"

echo "run-hostcall-second-pending-payload-revoke-probe.sh wrapper completed. Full serial log: $LOG_FILE"

