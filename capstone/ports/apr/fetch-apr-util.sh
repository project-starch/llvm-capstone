#!/usr/bin/env bash
# apr-util at a pinned version, checked, beside the APR fetch. The bucket
# allocator lives here and not in APR, which is why it needs its own fetch: the
# census of one is not the census of the other, and the two versions pair.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APU_VERSION=${APU_VERSION:-1.6.3}
ROOT=${CAPSTONE_TMP_ROOT:-/tmp/capstone}
DL=$ROOT/dl; mkdir -p "$DL"
TAR=$DL/apr-util-$APU_VERSION.tar.gz
SRC=$ROOT/apr-util-$APU_VERSION

if [ ! -f "$TAR" ]; then
  curl -sS -L -o "$TAR" "https://archive.apache.org/dist/apr/apr-util-$APU_VERSION.tar.gz"
fi
( cd "$DL" && sha256sum -c "$SCRIPT_DIR/sources.sha256" )
[ -d "$SRC" ] || tar xf "$TAR" -C "$ROOT"
echo "$SRC"
