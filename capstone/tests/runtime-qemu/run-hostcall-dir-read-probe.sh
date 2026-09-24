#!/usr/bin/env bash
# Directory listing (DIR_READ, getdents64) from a musl domain on the 9p share, with
# a control linked against a runtime without it. The test lives in dir-read/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/dir-read/run.sh" "$@"
