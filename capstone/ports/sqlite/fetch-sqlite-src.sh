#!/usr/bin/env bash
# Fetch and pin the FULL SQLite source tree, for the one file the amalgamation does not carry:
# test/speedtest1.c.
#
# WHY A SECOND FETCH. fetch-sqlite.sh gets the amalgamation zip, which is the engine only. The
# canonical benchmark ships in the full distribution. Both must be the SAME VERSION or the benchmark
# and the engine under test disagree about what they are measuring, so SQLITE_VERSION is shared and
# the check below is not optional.
#
# The tree was on this machine under /tmp before this script existed, which is one reboot from
# unreproducible -- the same reason the amalgamation is pinned rather than assumed.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

# Same version as fetch-sqlite.sh, expressed the way the full distribution names itself:
# the amalgamation is 3530300, the source tarball is 3.53.3.
SQLITE_VERSION=${SQLITE_VERSION:-3530300}
SQLITE_SRC_VERSION=${SQLITE_SRC_VERSION:-3.53.3}
SQLITE_SRC_SHA3=${SQLITE_SRC_SHA3:-03a50b977011695ca53aa6fb27e381816f9abfcbfd9e6214371432c1ad531716}
SQLITE_SRC_YEAR=${SQLITE_SRC_YEAR:-2026}
SQLITE_FULL_ROOT=${SQLITE_FULL_ROOT:-$CAPSTONE_TMP_ROOT/sqlite-src-full}
SQLITE_FULL_ARCHIVE=${SQLITE_FULL_ARCHIVE:-$SQLITE_FULL_ROOT/sqlite-$SQLITE_SRC_VERSION.tar.gz}
SQLITE_FULL_DIR=${SQLITE_FULL_DIR:-$SQLITE_FULL_ROOT/sqlite-version-$SQLITE_SRC_VERSION}
SQLITE_FULL_URL=${SQLITE_FULL_URL:-https://sqlite.org/src/tarball/sqlite.tar.gz?r=version-$SQLITE_SRC_VERSION}

mkdir -p "$SQLITE_FULL_ROOT"

if [[ ! -f "$SQLITE_FULL_ARCHIVE" ]]; then
  curl -L "$SQLITE_FULL_URL" -o "$SQLITE_FULL_ARCHIVE"
fi

python3 - "$SQLITE_FULL_ARCHIVE" "$SQLITE_SRC_SHA3" "$SQLITE_FULL_ROOT" "$SQLITE_FULL_DIR" <<'PY'
import hashlib, pathlib, sys, tarfile

archive = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
root = pathlib.Path(sys.argv[3])
src_dir = pathlib.Path(sys.argv[4])

actual = hashlib.sha3_256(archive.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(
        f"SQLite full-source SHA3-256 mismatch: expected {expected}, got {actual}"
    )

if not (src_dir / "test" / "speedtest1.c").is_file():
    with tarfile.open(archive) as source:
        source.extractall(root)
PY

# THE VERSION AGREEMENT CHECK, and it is the point of the file. The benchmark and the engine must
# come from the same release; a silent drift here would make every comparison a comparison of two
# SQLites. The full tree states its version in VERSION, the amalgamation in its own header.
if [[ -f "$SQLITE_FULL_DIR/VERSION" ]]; then
  _have=$(tr -d ' \n' < "$SQLITE_FULL_DIR/VERSION")
  [[ "$_have" == "$SQLITE_SRC_VERSION" ]] || {
    echo "ERROR: full source is $_have, expected $SQLITE_SRC_VERSION" >&2; exit 1; }
fi

for required in test/speedtest1.c VERSION; do
  [[ -e "$SQLITE_FULL_DIR/$required" ]] || {
    echo "ERROR: $SQLITE_FULL_DIR/$required missing after extract" >&2; exit 1; }
done

echo "$SQLITE_FULL_DIR"
