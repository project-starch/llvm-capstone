#!/usr/bin/env bash
# chdir() and getcwd() in a musl domain (a domain-side working directory), with a
# control on the runtime before them. Lives in cwd/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/cwd/run.sh" "$@"
