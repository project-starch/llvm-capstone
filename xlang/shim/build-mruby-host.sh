#!/usr/bin/env bash
# Build the xlang Phase-2 host for one mruby row.
#
#   ./build-mruby-host.sh ../4              # uses the row's host-asan build
#   ./build-mruby-host.sh ../4 host-asan
#
# Produces <row>/xlang-host, a drop-in replacement for
# <row>/mruby/build/<build>/bin/mruby that routes every VM allocation through
# the seam in mruby_host.c. See README.md.
set -euo pipefail

SHIM="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ $# -ge 1 ] || { echo "usage: $0 <row-dir> [build-name]" >&2; exit 2; }
ROW="$(cd "$1" && pwd)"
MRUBY="$ROW/mruby"

# Rows disagree on what they call the ASan build: 4, 5 and 10 use 'host-asan',
# the other nine use 'host'. Auto-detect so a caller only has to name the row.
BUILD="${2:-}"
if [ -z "$BUILD" ]; then
    for cand in host-asan host; do
        if [ -f "$MRUBY/build/$cand/lib/libmruby.a" ]; then
            BUILD="$cand"
            break
        fi
    done
fi
if [ -z "$BUILD" ]; then
    echo "error: no libmruby.a under $MRUBY/build/{host-asan,host}" >&2
    echo "       run $ROW/build.sh first" >&2
    exit 1
fi

LIB="$MRUBY/build/$BUILD/lib/libmruby.a"
[ -f "$LIB" ] || {
    echo "error: $LIB not found -- run $ROW/build.sh first" >&2
    exit 1
}

# The host must be instrumented exactly as the row's libmruby is, or ASan's
# interceptors and the library's poisoning disagree. These flags mirror the
# 'host-asan' target in every row's build_config.rb.
clang -fsanitize=address -g -O1 -fno-omit-frame-pointer \
      -I "$MRUBY/include" -I "$MRUBY/build/$BUILD/include" \
      -o "$ROW/xlang-host" \
      "$SHIM/mruby_host.c" "$LIB" -lm

echo "built $ROW/xlang-host"
