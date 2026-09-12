#!/bin/bash
# Compile PostgreSQL's memory manager for capstone64 and report what fails.
#
#   census-capstone.sh
#
# The census the SQLite and MicroPython ports run, for this one: it says what
# a freestanding capability build of the seven files costs, before anything is
# built for a domain. It reports rather than asserts, so a change in the
# compiler or in PostgreSQL shows up here as a different list and not as a
# failure somewhere downstream.
#
# What it needs: the Capstone clang. CAPSTONE_LLVM_BUILD_DIR if it is not at
# the default, and a PostgreSQL tree that has been configured once, which
# build-mmgr-host.sh leaves behind.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PG_VERSION=${PG_VERSION:-17.5}
OUT=${OUT:-${CAPSTONE_TMP_ROOT:-/tmp}/pg-mmgr-host}
SRC=$OUT/postgresql-$PG_VERSION
LLVM=${CAPSTONE_LLVM_BUILD_DIR:-$HERE/../../../llvm/build-rel}
CLANG=${CLANG:-$LLVM/bin/clang}
CENSUS=${CENSUS:-$OUT/census}

FILES="aset.c mcxt.c generation.c slab.c bump.c alignedalloc.c memdebug.c"

[ -x "$CLANG" ] || { echo "no Capstone clang at $CLANG; set CAPSTONE_LLVM_BUILD_DIR" >&2; exit 1; }
[ -f "$SRC/src/include/pg_config.h" ] || {
  echo "no configured PostgreSQL tree at $SRC; run build-mmgr-host.sh first" >&2; exit 1; }

mkdir -p "$CENSUS"
FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
       -ffreestanding -fno-builtin -O0
       -nostdlibinc -isystem "$HERE/port/stubinc"
       -I"$SRC/src/include" -I"$SRC/src/backend")

echo "== compiling the memory manager for capstone64"
ok=0; bad=0
for f in $FILES; do
  o=$CENSUS/${f%.c}.o
  rm -f "$o"
  src=$SRC/src/backend/utils/mmgr/$f
  # aset.c is compiled from a patched copy: two lines that a 16-byte pointer
  # forces, and the allocator itself says so, with a static assertion.
  if [ "$f" = aset.c ] && [ -f "$HERE/port/aset-capstone.patch" ]; then
    cp "$src" "$CENSUS/aset.c"
    patch -s -F0 -p0 "$CENSUS/aset.c" < "$HERE/port/aset-capstone.patch"
    src=$CENSUS/aset.c
  fi
  if "$CLANG" "${FLAGS[@]}" -c "$src" -o "$o" 2> "$CENSUS/${f%.c}.err"; then
    printf "  %-14s ok\n" "${f%.c}"; ok=$((ok + 1))
  else
    printf "  %-14s %s\n" "${f%.c}" "$(grep -m1 'error:' "$CENSUS/${f%.c}.err" | sed 's/.*error: //')"
    bad=$((bad + 1))
  fi
done
echo "  $ok of $((ok + bad)) objects"

echo
echo "== what the compiler says about capabilities"
cat "$CENSUS"/*.err | grep -E "capstone-pointer-roundtrip|capability" | sed 's/.*warning: //' \
  | sort -u | head -5 || true
echo "  (a round trip in an inline function the manager never calls is not a"
echo "   finding: check which file the line belongs to before acting on it)"

echo
echo "== what is undefined, once they are taken together"
"$LLVM/bin/llvm-nm" -u "$CENSUS"/*.o 2>/dev/null | awk '{print $2}' | sort -u \
  | grep -vE "^(AlignedAlloc|AllocSet|Bump|Generation|Slab|MemoryContext|GetMemoryChunk)" \
  | grep -v "^pfree$\|^TopMemoryContext$" | tr '\n' ' '
echo
