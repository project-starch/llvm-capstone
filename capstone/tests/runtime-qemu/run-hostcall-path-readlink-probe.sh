#!/usr/bin/env bash
# readlink() in a musl domain through PATH_READLINK (and realpath() on top of it), with a
# control on the runtime before it. Lives in path-readlink/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/path-readlink/run.sh" "$@"
