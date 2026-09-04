#!/usr/bin/env bash
set -euo pipefail

# Stage-2 SEALED-CALLBACK feasibility probe (callback-context revoke; rows 1/2/6/16).
# Uses EXISTING firmware ops only (shared_region_annotated REV_BORROWED +
# revoke_region + the already-sealed domain entry) -- no new monitor op, so it runs
# against the current rootfs.ext2/fw_jump.elf. The host registers a callback context,
# the engine invokes the sealed callback (round 1 reads pApp), the host unregisters
# (revoke), and round 2 re-invokes the callback.
# TRAPPED (round 2 == fault sentinel 0x0FA017ED) == the sealed callback invocation
# faulted on the revoked context == the shape composes from existing ops (no Step 2).
# NO-TRAP-GAP == a dedicated sealed-callback op is needed (Step 2).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
source "$SCRIPT_DIR/infra-retry.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-sqlite-sealed-callback-revoke-probe.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/sqlite_sealed_callback_revoke_probe.user \
      "$SHARE_DIR"/sqlite_sealed_callback_revoke_probe.smode

bash "$SCRIPT_DIR/build-sqlite-sealed-callback-revoke-probe.sh" "$SHARE_DIR"

capstone_retry_infra_flake \
  python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "cp /mnt/host/sqlite_sealed_callback_revoke_probe.user /tmp/scb.user && chmod 0755 /tmp/scb.user && /tmp/scb.user /mnt/host/sqlite_sealed_callback_revoke_probe.smode" \
  --success-marker "sqlite-sealed-cb: engine read callback context OK while registered" \
  --success-marker "sqlite-sealed-cb: callback unregistered (context revoked)" \
  --success-marker "sqlite-sealed-cb: round 2 returned"

echo "run-sqlite-sealed-callback-revoke-probe.sh completed. Full serial log: $LOG_FILE"
echo "Check round-2 line: TRAPPED == SEALED-CALLBACK composes from existing ops;"
echo "NO-TRAP-GAP == needs a dedicated sealed-callback op (Step 2)."
