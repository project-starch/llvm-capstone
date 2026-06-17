#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=nettle-arcfour
BEEBS_SOURCE_FILES_REL=(src/nettle-arcfour/arcfour.c)
BEEBS_STRIP_HOSTED_INCLUDES=1
BEEBS_STRIP_EXTRA_HEADERS=(stdint)
BEEBS_PREAMBLE_LINES=('typedef unsigned char uint8_t;')
# verify_benchmark has "int exp[16]" as a local array; at -O0 the backend uses
# stc (16-byte capability store) to copy it from .rodata, zeroing bytes 8-15 of
# each 16-byte stc chunk, which corrupts exp[2..3], exp[6..7], etc.
# Fix: make it static so it lives in .rodata and needs no runtime copy.
BEEBS_EXTRA_SED_EXPRS=('s/  int exp\[\]/  static int exp[]/')
source "$SCRIPT_DIR/build-beebs-simple-capstone-common.sh"
