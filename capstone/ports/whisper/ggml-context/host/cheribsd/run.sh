#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then
  echo "usage: bash host/cheribsd/run.sh BUILD OUTPUT --sdk SDK --rootfs ROOTFS --image IMAGE" >&2
  exit 2
fi
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD=$1
OUTPUT=$2
shift 2
exec "${PYTHON:-python3}" "$HERE/../../../../common/host/cheribsd/run.py" "$OUTPUT" --build "whisper=$BUILD" "$@"
