#!/usr/bin/env bash
# __thread in a musl domain (C-47): local-exec codegen, the TLS segment, and the runtime's block; with an overrun control and an old-runtime control. Lives in thread-local/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/thread-local/run.sh" "$@"
