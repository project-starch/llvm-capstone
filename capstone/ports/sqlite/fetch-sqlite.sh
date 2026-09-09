#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

SQLITE_VERSION=${SQLITE_VERSION:-3530300}
SQLITE_YEAR=${SQLITE_YEAR:-2026}
SQLITE_ARCHIVE_SHA3=${SQLITE_ARCHIVE_SHA3:-d45c688a8cb23f68611a894a756a12d7eb6ab6e9e2468ca70adbeab3808b5ab9}
SQLITE_BASENAME="sqlite-amalgamation-$SQLITE_VERSION"
SQLITE_CACHE_ROOT=${SQLITE_CACHE_ROOT:-$CAPSTONE_TMP_ROOT/sqlite-src}
SQLITE_ARCHIVE=${SQLITE_ARCHIVE:-$SQLITE_CACHE_ROOT/$SQLITE_BASENAME.zip}
SQLITE_SRC_DIR=${SQLITE_SRC_DIR:-$SQLITE_CACHE_ROOT/$SQLITE_BASENAME}
SQLITE_URL=${SQLITE_URL:-https://www.sqlite.org/$SQLITE_YEAR/$SQLITE_BASENAME.zip}

mkdir -p "$SQLITE_CACHE_ROOT"

if [[ ! -f "$SQLITE_ARCHIVE" ]]; then
  curl -L "$SQLITE_URL" -o "$SQLITE_ARCHIVE"
fi

python3 - "$SQLITE_ARCHIVE" "$SQLITE_ARCHIVE_SHA3" "$SQLITE_CACHE_ROOT" "$SQLITE_SRC_DIR" <<'PY'
import hashlib
import pathlib
import sys
import zipfile

archive = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
cache_root = pathlib.Path(sys.argv[3])
src_dir = pathlib.Path(sys.argv[4])

actual = hashlib.sha3_256(archive.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(
        f"SQLite archive SHA3-256 mismatch: expected {expected}, got {actual}"
    )

if not (src_dir / "sqlite3.c").is_file() or not (src_dir / "sqlite3.h").is_file():
    with zipfile.ZipFile(archive) as source:
        source.extractall(cache_root)
PY

for required in sqlite3.c sqlite3.h; do
  [[ -f "$SQLITE_SRC_DIR/$required" ]] || {
    echo "missing SQLite amalgamation file: $SQLITE_SRC_DIR/$required" >&2
    exit 1
  }
done

printf '%s\n' "$SQLITE_SRC_DIR"
