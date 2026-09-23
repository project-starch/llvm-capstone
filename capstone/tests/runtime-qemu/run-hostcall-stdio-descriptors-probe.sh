#!/usr/bin/env bash
# The standard descriptors of a musl domain through every syscall that takes a
# descriptor, with a control on a runtime whose fstat knows no stdout. Lives in
# stdio-descriptors/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/stdio-descriptors/run.sh" "$@"
