#!/usr/bin/env bash
set -euo pipefail

# Stage-2 hierarchical-cascade feasibility experiment (use-after-close rows).
# Uses ONLY existing lender ops (no firmware change): the engine lends a parent
# (connection) and a child (statement value) region, the host caches the child,
# the engine revokes the PARENT (= sqlite3_close), and round 2 re-reads the child.
# If the parent revoke cascades, the round-2 read is TRAPPED. Otherwise the child
# read succeeds and a faithful cascade needs a derived-child monitor extension.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
source "$SCRIPT_DIR/infra-retry.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-sqlite-hier-revoke-probe.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/sqlite_hier_revoke_probe.user \
      "$SHARE_DIR"/sqlite_hier_revoke_probe.smode

bash "$SCRIPT_DIR/build-sqlite-hier-revoke-probe.sh" "$SHARE_DIR"

capstone_retry_infra_flake \
  python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "cp /mnt/host/sqlite_hier_revoke_probe.user /tmp/shr.user && chmod 0755 /tmp/shr.user && /tmp/shr.user /mnt/host/sqlite_hier_revoke_probe.smode" \
  --success-marker "sqlite-hier-revoke: host read statement value OK before close" \
  --success-marker "sqlite-hier-revoke: close revoked the connection (parent)" \
  --success-marker "sqlite-hier-revoke: round 2 returned"

echo "run-sqlite-hier-revoke-probe.sh completed. Full serial log: $LOG_FILE"
echo "Check round-2 line: TRAPPED == hierarchical cascade works with existing ops;"
echo "NO-CASCADE == needs a derived-child monitor extension."
