#!/usr/bin/env bash
# Build the NATIVE SQLLogicTest baseline binary.
#
# Same runner header and same SQLite amalgamation as the domain build, compiled for the
# build machine. Produces the numbers every domain run is compared against; see
# slt/slt_native.c for why the comparison and not the absolute rate is the result.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/slt-native}
mkdir -p "$OUT_DIR"

# The SAME amalgamation the domain gets: fetch-sqlite.sh verifies its SHA3 and prints the
# directory. Using a system libsqlite3 instead would silently baseline a different engine.
SQLITE_SRC_DIR=$(bash "$SCRIPT_DIR/fetch-sqlite.sh")
echo "== amalgamation: $SQLITE_SRC_DIR"

# THREADSAFE=0 and the memory-VFS-friendly defaults mirror the domain build's shape closely
# enough for a fair comparison, but they are NOT the reason a difference would appear: the
# runner uses only the public API, so a divergence points at code generation, not config.
cc -O1 -o "$OUT_DIR/slt_native" \
  -I"$SQLITE_SRC_DIR" -I"$SCRIPT_DIR/slt" \
  -DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION \
  "$SCRIPT_DIR/slt/slt_native.c" "$SQLITE_SRC_DIR/sqlite3.c" \
  -lm

echo "Built $OUT_DIR/slt_native"
