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
# verify_benchmark has "uint8_t expected[16]" as a local array; same stc-copy
# issue as nettle-arcfour. Fix: make it static.
BEEBS_EXTRA_SED_EXPRS=('s/  uint8_t expected\[16\]/  static uint8_t expected[16]/')
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
