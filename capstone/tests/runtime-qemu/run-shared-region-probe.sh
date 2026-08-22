#!/usr/bin/env bash
set -euo pipefail

# One-command regression wrapper for the restored shared-region sentinel proof.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
source "$SCRIPT_DIR/infra-retry.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-shared-region-probe.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/shared_region_probe.user "$SHARE_DIR"/shared_region_probe.smode

# Rebuild the probe into the host-shared 9p directory exported to the guest.
bash "$SCRIPT_DIR/build-shared-region-probe.sh" "$SHARE_DIR"

# The wrapper is green only if the helper observes both shared-memory mutations.
capstone_retry_infra_flake \
  python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "cp /mnt/host/shared_region_probe.user /tmp/shared_region_probe.user && chmod 0755 /tmp/shared_region_probe.user && /tmp/shared_region_probe.user /mnt/host/shared_region_probe.smode" \
  --success-marker "shared-region-probe: word after call 1 = 0x1111111111111111" \
  --success-marker "shared-region-probe: word after call 2 = 0x2222222222222222" \
  --success-marker "shared-region-probe: success"

echo "run-shared-region-probe.sh wrapper completed. Full serial log: $LOG_FILE"

