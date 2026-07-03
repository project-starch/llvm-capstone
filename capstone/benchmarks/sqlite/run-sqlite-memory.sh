#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/sqlite-build}
SHARE_DIR=${SHARE_DIR:-$OUT_DIR}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-sqlite-memory.log}

mkdir -p "$OUT_DIR" "$SHARE_DIR"
rm -f "$SHARE_DIR/sqlite_memory_capstone.dom" "$SHARE_DIR/sqlite_host.user"

OUT_DIR="$OUT_DIR" OUT_DOM="$SHARE_DIR/sqlite_memory_capstone.dom" \
  bash "$SCRIPT_DIR/build-sqlite-capstone.sh"
OUT_DIR="$OUT_DIR" OUT_HOST="$SHARE_DIR/sqlite_host.user" \
  bash "$SCRIPT_DIR/build-sqlite-host.sh"

python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier 6 \
  --guest-command \
    'cp /mnt/host/sqlite_host.user /tmp/sqlite_host.user && chmod 0755 /tmp/sqlite_host.user && /tmp/sqlite_host.user /mnt/host/sqlite_memory_capstone.dom' \
  --success-marker 'row name=alpha value=11' \
  --success-marker 'row name=beta value=22' \
  --success-marker 'row name=gamma value=33' \
  --success-marker '__CAPSTONE_SQLITE_EXTENDED_PASSED__' \
  --success-marker '__CAPSTONE_SQLITE_MEMORY_PASSED__'

echo "__CAPSTONE_SQLITE_MEMORY_PASSED__"
echo "run-sqlite-memory.sh completed. Full serial log: $LOG_FILE"
