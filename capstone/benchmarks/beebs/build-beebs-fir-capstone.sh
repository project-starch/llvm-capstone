#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=fir
BEEBS_SOURCE_FILES_REL=(src/fir/libfir.c)
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
