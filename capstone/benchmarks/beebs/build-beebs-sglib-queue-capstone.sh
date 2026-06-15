#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=sglib-queue
BEEBS_SOURCE_FILES_REL=(src/sglib-queue/queue.c)
BEEBS_EXTRA_INCLUDE_RELS=(src/sglib-queue)
BEEBS_STRIP_HOSTED_INCLUDES=1
BEEBS_DEFINE_NULL=1
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
