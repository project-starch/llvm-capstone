#!/usr/bin/env bash
# Does ngx_palloc.c build freestanding for capstone64 at all, and what does it then want?
# A7's table names one risk for this allocator, "the header chain is the whole server", and this
# answers it rather than estimating around it. No domain, no run: one compile.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/fetch-nginx.sh" >/dev/null
NGX_SRC_DIR=${NGX_SRC_DIR:-/tmp/capstone/nginx-${NGX_VERSION:-1.28.0}}
OUT=${OUT_DIR:-/tmp/capstone/nginx-census}; mkdir -p "$OUT"
CLANG=${CAPSTONE_CLANG:-${CAPSTONE_LLVM_BUILD_DIR:-/home/biecho/llvm-capstone/llvm/build-rel}/bin/clang}
NM=${CAPSTONE_LLVM_NM:-$(dirname "$CLANG")/llvm-nm}

# The allocator and its header, with ngx_config.h and ngx_core.h replaced by the shim beside this
# script. Everything else is upstream, byte for byte.
cp "$NGX_SRC_DIR/src/core/ngx_palloc.c" "$NGX_SRC_DIR/src/core/ngx_palloc.h" "$OUT/"
cp "$SCRIPT_DIR/adapted/ngx_shim.h" "$OUT/"
sed -i 's|#include <ngx_config.h>|#include "ngx_shim.h"|; s|#include <ngx_core.h>||' "$OUT/ngx_palloc.c"
sed -i 's|#include <ngx_config.h>||; s|#include <ngx_core.h>||' "$OUT/ngx_palloc.h"

for o in 0 1 2; do
  "$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
      -mllvm -capstone-gp-captable -ffreestanding -fno-jump-tables -std=c99 -w \
      -O$o -c -o "$OUT/ngx_palloc-O$o.o" "$OUT/ngx_palloc.c" \
    && printf "  -O%s   ok, %s bytes\n" "$o" "$(stat -c %s "$OUT/ngx_palloc-O$o.o")" \
    || { echo "  -O$o   FAILED" >&2; exit 1; }
done

echo "what it still wants from a level below:"
"$NM" --undefined-only "$OUT/ngx_palloc-O0.o" | awk '{printf "  %s\n", $2}'
echo "the shim that replaced ngx_core.h's 47 headers: $(wc -l < "$SCRIPT_DIR/adapted/ngx_shim.h") lines"
