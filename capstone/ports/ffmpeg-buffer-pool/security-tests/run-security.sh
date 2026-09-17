#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh"
export FFPOOL_WORK=${FFPOOL_WORK:-$CAPSTONE_TMP_ROOT/ffmpeg-buffer-pool}
export CAPSTONE_REV_NODES=${CAPSTONE_REV_NODES:-1048576}
flock -x -w 45 "$CAPSTONE_QEMU_LOCK" "${PYTHON:-python3}" "$HERE/security-suite.py" "$@"
