#!/usr/bin/env bash
# mkdir() and rmdir() in a musl domain through PATH_MKDIR and PATH_DELETE's directory
# flag, with a control on the runtime before them. Lives in mkdir-rmdir/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/mkdir-rmdir/run.sh" "$@"
