#!/usr/bin/env bash
# Native functional comparison; optional mode numbers do not enable revocation.
set -euo pipefail

WORK=${FFPOOL_WORK:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ffmpeg-buffer-pool}

BUILD_DIR=${FFPOOL_NATIVE_BUILD_DIR:-$WORK/build/native}

exec "$BUILD_DIR/bin/replay" "$@"
