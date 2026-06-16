#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=sglib-hashtable
BEEBS_SOURCE_FILES_REL=(src/sglib-hashtable/hashtable.c)
BEEBS_EXTRA_INCLUDE_RELS=(src/sglib-hashtable)
BEEBS_DEFINE_NULL=1
BEEBS_STRIP_HOSTED_INCLUDES=1
BEEBS_STRIP_EXTRA_HEADERS=(string)
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
