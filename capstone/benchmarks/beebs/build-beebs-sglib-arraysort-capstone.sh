#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=sglib-arraysort
BEEBS_SOURCE_FILES_REL=(src/sglib-arraysort/arraysort.c)
BEEBS_EXTRA_INCLUDE_RELS=(src/sglib-arraysort)
BEEBS_EXTRA_DEFINES=(QUICK_SORT)
BEEBS_DEFINE_NULL=1
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
