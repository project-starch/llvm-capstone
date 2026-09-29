#!/usr/bin/env bash
# The runtime's unserved-syscall report survives a program that closed fd 1, with a control
# on the runtime before ISSUES I-11 (the report was refused and lost). Lives in unserved-report/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/unserved-report/run.sh" "$@"
