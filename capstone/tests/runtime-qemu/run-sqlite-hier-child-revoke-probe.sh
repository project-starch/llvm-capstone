#!/usr/bin/env bash
set -euo pipefail

# Stage-2 hierarchical derived-child cascade probe (use-after-close rows).
# Requires the firmware built with the share_child_region monitor op (fid 0xd) and
# the modcapstone IOCTL_REGION_SHARE_CHILD -- i.e. a rootfs.ext2/fw_jump.elf rebuilt
# after adding that op. The engine writes a column value INSIDE the connection,
# lends a child sub-window DERIVED from the connection, the host caches it, the
# engine revokes the PARENT (= sqlite3_close), and round 2 re-reads the child.
# TRAPPED (round 2 == fault sentinel 0x0FA017ED) == the parent revoke cascaded to
# the derived child == use-after-close defeated.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-sqlite-hier-child-revoke-probe.log}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/sqlite_hier_child_revoke_probe.user \
      "$SHARE_DIR"/sqlite_hier_child_revoke_probe.smode

bash "$SCRIPT_DIR/build-sqlite-hier-child-revoke-probe.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --guest-command "cp /mnt/host/sqlite_hier_child_revoke_probe.user /tmp/shr.user && chmod 0755 /tmp/shr.user && /tmp/shr.user /mnt/host/sqlite_hier_child_revoke_probe.smode" \
  --success-marker "sqlite-hier-child: host read statement value OK before close" \
  --success-marker "sqlite-hier-child: close revoked the connection (parent)" \
  --success-marker "sqlite-hier-child: round 2 returned"

echo "run-sqlite-hier-child-revoke-probe.sh completed. Full serial log: $LOG_FILE"
echo "Check round-2 line: TRAPPED == hierarchical cascade works (derived child);"
echo "NO-CASCADE-GAP == the child was not invalidated by the parent revoke."
