#!/usr/bin/env bash
# Constructors and destructors in a musl domain, with a control on the runtime before
# ISSUES C-64 (no constructor ran; exit() faulted on .fini_array). Lives in init-fini/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/init-fini/run.sh" "$@"
