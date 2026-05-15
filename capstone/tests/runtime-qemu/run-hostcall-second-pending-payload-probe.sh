#!/usr/bin/env bash
set -euo pipefail

# One-command diagnostic wrapper for the narrower question:
# can the same borrowed output payload region be reused across two successive
# PENDING rounds, or does that reproduce the current helper_csmrev assertion?

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-hostcall-second-pending-payload-probe.log}
WRAPPER_LOG=${WRAPPER_LOG:-$TMP_ROOT/capstone-runtime-qemu-hostcall-second-pending-payload-probe-wrapper.txt}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/hostcall_second_pending_payload_probe.user \
      "$SHARE_DIR"/hostcall_second_pending_payload_probe.smode

bash "$SCRIPT_DIR/build-hostcall-second-pending-payload-probe.sh" "$SHARE_DIR"

set +e
python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "cp /mnt/host/hostcall_second_pending_payload_probe.user /tmp/hostcall_second_pending_payload_probe.user && chmod 0755 /tmp/hostcall_second_pending_payload_probe.user && /tmp/hostcall_second_pending_payload_probe.user /mnt/host/hostcall_second_pending_payload_probe.smode" \
  --success-marker "hostcall-second-pending-payload-probe: second pending with payload reuse observed" \
  --success-marker "hostcall-second-pending-payload-probe: success" \
  --success-marker "__HOSTCALL_SECOND_PENDING_PAYLOAD_OK__" \
  > "$WRAPPER_LOG" 2>&1
status=$?
set -e

if [ "$status" -eq 0 ]; then
  echo "hostcall-second-pending-payload-probe: payload reuse across a second PENDING is supported in this environment"
  echo "run-hostcall-second-pending-payload-probe.sh wrapper completed. Full serial log: $LOG_FILE"
  exit 0
fi

if grep -q "helper_csmrev: Assertion .*CAP_TYPE_LIN" "$WRAPPER_LOG" "$LOG_FILE"; then
  echo "hostcall-second-pending-payload-probe: reproduced current helper_csmrev assertion during payload-reuse second-PENDING attempt"
  echo "__HOSTCALL_SECOND_PENDING_PAYLOAD_ASSERT__"
  echo "run-hostcall-second-pending-payload-probe.sh wrapper completed. Full serial log: $LOG_FILE"
  exit 0
fi

echo "hostcall-second-pending-payload-probe: unexpected failure; inspect logs:" >&2
echo "  wrapper: $WRAPPER_LOG" >&2
echo "  serial:  $LOG_FILE" >&2
exit 1

