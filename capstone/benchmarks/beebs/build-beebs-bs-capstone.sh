#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=bs
BEEBS_SOURCE_FILES_REL=(src/bs/libbs.c)
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
