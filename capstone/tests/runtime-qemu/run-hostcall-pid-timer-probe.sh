#!/usr/bin/env bash
# getpid, getppid, umask and the never-firing setitimer/getitimer in a musl domain,
# with a control on the runtime before them. Lives in pid-timer/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/pid-timer/run.sh" "$@"
