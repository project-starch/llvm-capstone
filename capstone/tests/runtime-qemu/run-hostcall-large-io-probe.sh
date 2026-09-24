#!/usr/bin/env bash
# Large file reads and writes on the 9p share, and a stdout line longer than a
# payload region with the host's stdout on a 9p file, with a pinned control for
# each direction. The test and its controls live in large-io/; see its run.sh.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/large-io/run.sh" "$@"
