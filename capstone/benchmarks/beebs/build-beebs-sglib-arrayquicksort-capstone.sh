#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=sglib-arrayquicksort
BEEBS_SOURCE_FILES_REL=(src/sglib-arrayquicksort/arraysort.c)
BEEBS_EXTRA_INCLUDE_RELS=(src/sglib-arrayquicksort)
BEEBS_EXTRA_DEFINES=(QUICK_SORT)
BEEBS_DEFINE_NULL=1
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
