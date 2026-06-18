#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=sglib-rbtree
BEEBS_SOURCE_FILES_REL=(src/sglib-rbtree/rbtree.c)
BEEBS_EXTRA_INCLUDE_RELS=(src/sglib-rbtree)
BEEBS_STRIP_HOSTED_INCLUDES=1
BEEBS_DEFINE_NULL=1
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
