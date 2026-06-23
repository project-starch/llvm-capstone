#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=janne_complex
BEEBS_SOURCE_FILES_REL=(src/janne_complex/libjanne_complex.c)
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
