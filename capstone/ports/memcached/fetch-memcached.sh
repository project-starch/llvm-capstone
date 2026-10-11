#!/usr/bin/env bash
# memcached at a pinned version, checked, into the shared tmp root. The same shape as
# fetch-apr.sh and fetch-nginx.sh, so a census of one target can be read against another.
#
# The hash was recorded from memcached.org on 2026-09-21; it is the pin, not a claim about
# the upstream release process. A second source has not been checked against it.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MC_VERSION=${MC_VERSION:-1.6.45}
ROOT=${CAPSTONE_TMP_ROOT:-/tmp/capstone}
DL=$ROOT/dl; mkdir -p "$DL"
TAR=$DL/memcached-$MC_VERSION.tar.gz
SRC=$ROOT/memcached-$MC_VERSION

if [ ! -f "$TAR" ]; then
  curl -sS -L -o "$TAR" "https://memcached.org/files/memcached-$MC_VERSION.tar.gz"
fi
( cd "$DL" && sha256sum -c "$SCRIPT_DIR/sources.sha256" )
[ -d "$SRC" ] || tar xf "$TAR" -C "$ROOT"
echo "$SRC"
