#!/usr/bin/env bash
# pipe2, a pipe's read/write/fcntl/fstat/lseek, and poll in a musl domain (the self-pipe
# latch), with a control on the runtime before them. Lives in pipe-poll/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/pipe-poll/run.sh" "$@"
