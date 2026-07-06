#!/usr/bin/env bash
set -euo pipefail

# Stage-2 "after" for cve-repros/row3_diesel_colname_cached: a SQLite column
# pointer cached across sqlite3_step (diesel RUSTSEC-2021-0037), enforced by
# monitor-mediated revocation (the #70 path).
#
# The engine lends the current row buffer as a REV_BORROWED region; the host
# reads the column (round 1) and caches the pointer; the engine "steps"
# (revoke_region); the host re-reads the cached pointer (round 2 = the UAF).
# With revocation enforced the cached cap reloads untagged, the read faults, and
# the monitor cleanly terminates the domain -- the engine sees the fault sentinel
# (0x0FA017ED) in round 2 instead of the stale column value. Safe-fail: the
# use-after-free is a deterministic trap.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-sqlite-borrow-revoke-probe.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/sqlite_borrow_revoke_probe.user \
      "$SHARE_DIR"/sqlite_borrow_revoke_probe.smode

bash "$SCRIPT_DIR/build-sqlite-borrow-revoke-probe.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "cp /mnt/host/sqlite_borrow_revoke_probe.user /tmp/sbr.user && chmod 0755 /tmp/sbr.user && /tmp/sbr.user /mnt/host/sqlite_borrow_revoke_probe.smode" \
  --success-marker "sqlite-borrow-revoke: host read column OK before step" \
  --success-marker "sqlite-borrow-revoke: step revoked the column borrow" \
  --success-marker "sqlite-borrow-revoke: use-after-free read TRAPPED"

echo "run-sqlite-borrow-revoke-probe.sh completed. Full serial log: $LOG_FILE"
echo "ROW-3 AFTER: a column pointer cached across sqlite3_step (diesel"
echo "RUSTSEC-2021-0037) is TRAPPED by monitor-mediated revocation (safe-fail)."
