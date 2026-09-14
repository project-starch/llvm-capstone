#!/usr/bin/env bash
# APR at a pinned version, checked, into the shared tmp root. The same shape as fetch-nginx.sh so
# that a census of one target can be read against a census of the other.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APR_VERSION=${APR_VERSION:-1.7.4}
ROOT=${CAPSTONE_TMP_ROOT:-/tmp/capstone}
DL=$ROOT/dl; mkdir -p "$DL"
TAR=$DL/apr-$APR_VERSION.tar.gz
SRC=$ROOT/apr-$APR_VERSION

if [ ! -f "$TAR" ]; then
  curl -sS -L -o "$TAR" "https://archive.apache.org/dist/apr/apr-$APR_VERSION.tar.gz"
fi
( cd "$DL" && sha256sum -c "$SCRIPT_DIR/sources.sha256" )
[ -d "$SRC" ] || tar xf "$TAR" -C "$ROOT"
echo "$SRC"
