#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=sglib-arrayheapsort
BEEBS_SOURCE_FILES_REL=(src/sglib-arrayheapsort/arraysort.c)
BEEBS_EXTRA_INCLUDE_RELS=(src/sglib-arrayheapsort)
BEEBS_EXTRA_DEFINES=(HEAP_SORT)
BEEBS_DEFINE_NULL=1
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
