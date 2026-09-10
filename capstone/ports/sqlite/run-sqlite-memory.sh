#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/sqlite-build}

# Q-01 (2026-09-05): the -O0 amalgamation with the 1 MB arena made a 3.56 MB image (LOAD memsz
# 0x365bb0: .text 2.44 MB, .bss 1.05 MB), and the kernel module allocates code_len + max(code_len,
# 64 KiB) -- ~7 MB, order 11, above the kernel's MAX_ORDER 10 -- so create_dom failed before any
# SQL ran, every night, since the module matched the board's geometry (37ed834). The silicon arm
# (1.38 MB, 256 KB arena) fits. So this arm builds the amalgamation at -O1 with the 256 KB arena
# and keeps everything else (default ABI, -O0 glue/libc/VFS, the extended workload). -O0 SQLite
# coverage is not lost by this: the SLT twins run -O0 arms. Override with SQLITE_OPT_LEVEL /
# DOMAIN_EXTRA_FLAGS if a bigger image is wanted, and expect Q-01 back above ~2 MB memsz.
export SQLITE_OPT_LEVEL=${SQLITE_OPT_LEVEL:--O1}
export DOMAIN_EXTRA_FLAGS="-DSQLITE_HEAP_SIZE=262144 ${DOMAIN_EXTRA_FLAGS:-}"
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
