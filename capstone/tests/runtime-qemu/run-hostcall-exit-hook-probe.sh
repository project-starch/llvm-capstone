#!/usr/bin/env bash
# exit() from a musl domain, with and without a __capstone_at_exit hook, and a
# control on a runtime that tests the hook's address (C-56). Lives in exit-hook/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/exit-hook/run.sh" "$@"
