#!/bin/bash
# PostgreSQL's memory manager, outside PostgreSQL, on the host.
#
#   build-mmgr-host.sh [<trace> ...]
#
# Builds the seven files of src/backend/utils/mmgr against pg_stubs.c and the
# replay driver, then replays each trace given. With no trace it builds and
# stops, which is the gate that says the manager still stands on its own.
#
# This is the host arm. The freestanding arm builds the same seven files and
# the same driver for a domain; what differs is the compiler, the level below,
# and the three printf helpers. Keeping the host arm working is what makes a
# fault in the freestanding one a fault about capabilities.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$HERE/../../.." && pwd)
PG_VERSION=${PG_VERSION:-17.5}
BASE_URL=https://ftp.postgresql.org/pub/source/v$PG_VERSION
OUT=${OUT:-${CAPSTONE_TMP_ROOT:-/tmp}/pg-mmgr-host}
SRC=$OUT/postgresql-$PG_VERSION

# The seven files, and the count is the claim: this is the whole memory
# manager, and nothing else of the backend is here.
FILES="aset.c mcxt.c generation.c slab.c bump.c alignedalloc.c memdebug.c"

mkdir -p "$OUT"
if [ ! -f "$SRC/configure" ]; then
  echo "== fetching PostgreSQL $PG_VERSION"
  curl -sS -L -o "$OUT/pg.tar.bz2" "$BASE_URL/postgresql-$PG_VERSION.tar.bz2"
  tar xf "$OUT/pg.tar.bz2" -C "$OUT"
fi
# configure only generates the headers the manager includes; nothing of the
# backend is built, and the manager is compiled from the tree by hand below.
if [ ! -f "$SRC/src/include/pg_config.h" ]; then
  echo "== configuring, for its generated headers only"
  (cd "$SRC" && ./configure --without-icu --without-zlib --without-readline \
       --without-libxml --quiet)
  make -C "$SRC/src" submake-generated-headers >/dev/null
fi

CC=${CC:-cc}
# a11trace.h is here, beside the replay, because a domain build has to be
# self-contained; the recorder in the paper fetches it from here at a pinned
# commit.
INC="-I$SRC/src/include -I$SRC/src/backend -I$HERE/port -I$HERE"

echo "== building the manager, $(cd "$SRC/src/backend/utils/mmgr" && wc -l $FILES | tail -1 | awk '{print $1}') lines"
mkdir -p "$OUT/obj"
for f in $FILES; do
  $CC -c -O2 -g $INC "$SRC/src/backend/utils/mmgr/$f" -o "$OUT/obj/${f%.c}.o"
done
$CC -c -O2 -g $INC "$HERE/port/pg_stubs.c" -o "$OUT/obj/pg_stubs.o"
$CC -O2 -g $INC "$HERE/tools/replay.c" "$OUT/obj"/*.o -o "$OUT/replay"
echo "== the manager links against $(grep -c '^[A-Za-z].*(' "$HERE/port/pg_stubs.c") definitions and libc"
echo "   $OUT/replay"

for t in "$@"; do
  echo
  echo "== replaying $(basename "$t")"
  "$OUT/replay" "$t"
done
