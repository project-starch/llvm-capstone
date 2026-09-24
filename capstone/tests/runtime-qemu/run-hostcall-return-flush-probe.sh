#!/usr/bin/env bash
# A musl domain that returns from its entry without exit(): buffered stdout and atexit
# handlers must still reach the host, with a control on a runtime that returns straight
# to the host. Lives in return-flush/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/return-flush/run.sh" "$@"
