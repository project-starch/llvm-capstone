#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=sglib-arraybinsearch
BEEBS_SOURCE_FILES_REL=(src/sglib-arraybinsearch/arraybinsearch.c)
BEEBS_EXTRA_INCLUDE_RELS=(src/sglib-arraybinsearch)
BEEBS_STRIP_HOSTED_INCLUDES=1
BEEBS_DEFINE_NULL=1
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
