#!/usr/bin/env bash
set -euo pipefail

# Build and run the first Capstone-compiled SQLite-facing VFS skeleton domain.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-sqlite-vfs-skeleton.log}
DOMAIN_PATH="$SHARE_DIR/sqlite_vfs_skeleton.dom"

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$DOMAIN_PATH"

bash "$SCRIPT_DIR/build-sqlite-vfs-skeleton.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  "$DOMAIN_PATH"

echo "run-sqlite-vfs-skeleton.sh wrapper completed. Full serial log: $LOG_FILE"

