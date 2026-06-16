#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=nsichneu
BEEBS_SOURCE_FILES_REL=(src/nsichneu/libnsichneu.c)
BEEBS_STRIP_HOSTED_INCLUDES=1
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
