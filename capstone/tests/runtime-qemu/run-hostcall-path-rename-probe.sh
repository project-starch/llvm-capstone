#!/usr/bin/env bash
# rename() in a musl domain through PATH_RENAME, with a control on the runtime before
# it. Lives in path-rename/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/path-rename/run.sh" "$@"
