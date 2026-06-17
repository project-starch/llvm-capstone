#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=nettle-des
BEEBS_SOURCE_FILES_REL=(src/nettle-des/des.c)
BEEBS_STRIP_HOSTED_INCLUDES=1
BEEBS_STRIP_EXTRA_HEADERS=(stdint)
BEEBS_PREAMBLE_LINES=(
  'typedef unsigned char uint8_t;'
  'typedef signed char int8_t;'
  'typedef unsigned int uint32_t;'
)
BEEBS_EXTRA_INCLUDE_RELS=(src/nettle-des)
BEEBS_STRIP_FROM_REGEX='^int verify_benchmark'
BEEBS_ADAPTED_TAIL_SRC=$SCRIPT_DIR/adapted/beebs_nettle_des_capstone_tail.c
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
