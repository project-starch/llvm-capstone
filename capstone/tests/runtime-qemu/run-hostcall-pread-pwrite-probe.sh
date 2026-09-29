#!/usr/bin/env bash
# pread, pwrite, preadv and pwritev in a musl domain, with a control on the runtime
# before them. Lives in pread-pwrite/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/pread-pwrite/run.sh" "$@"
