#!/usr/bin/env bash
# mmap, munmap and System V shared memory in a musl domain, served from level0, with a
# control linked without the override. Lives in mmap-shm/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/mmap-shm/run.sh" "$@"
