#!/usr/bin/env bash
# Run SQLite 3.22.0 through the silicon-config QEMU gate.
#
# The gate is run-sqlite-silicon.sh, unchanged: the same domain glue, host, VFS and the same five
# success markers as 3.53.3. Only the amalgamation it compiles differs. This script fetches it
# (pinned by SHA3-256), adapts it with adapt-sqlite-322.sh, and hands it over via PATCHED_SQLITE.
#
# THE SCRATCH ROOT IS SEPARATE ON PURPOSE. build-sqlite-silicon.sh stages sqlite3.h from the first
# match of $CAPSTONE_TMP_ROOT/sqlite-src/sqlite-amalgamation-*/sqlite3.h, so a root holding both
# versions would compile one version's header against the other's amalgamation. It is set
# unconditionally for that reason; override with SQLITE322_TMP_ROOT, never with the 3.53.3 root.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CAPSTONE_TMP_ROOT=${SQLITE322_TMP_ROOT:-/tmp/capstone-322}
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

export SQLITE_VERSION=3220000
export SQLITE_YEAR=2018
export SQLITE_ARCHIVE_SHA3=69bc5ee8f08d747494dd3a4bfe075e5b078fe200dfc671d76dd9e1ccb5b2decb
SRC=$(bash "$SCRIPT_DIR/fetch-sqlite.sh")

# Regenerated on every run, never reused: build-sqlite-silicon.sh compiles whatever file sits here.
export PATCHED_SQLITE=$CAPSTONE_TMP_ROOT/sqlite-build/sqlite3-capstone.c
mkdir -p "$(dirname -- "$PATCHED_SQLITE")"
bash "$SCRIPT_DIR/adapt-sqlite-322.sh" "$SRC/sqlite3.c" "$PATCHED_SQLITE"

exec bash "$SCRIPT_DIR/run-sqlite-silicon.sh"
