#!/usr/bin/env bash
# The native oracle: the same SQLite 3.22.0 release, unadapted, built for the host with its
# defaults. The Capstone build's results are compared against these programs'.
#   usage: build-native.sh <prepared sources (prepare-sources.sh)> <out dir>
set -euo pipefail
SRC=${1:?usage: build-native.sh <sources> <out>}
OUT=${2:?usage: build-native.sh <sources> <out>}
CC=${CC:-cc}
CONFIG=(-DHAVE_MALLOC_H=1 -DHAVE_MALLOC_USABLE_SIZE=1)   # as build-domain.sh
mkdir -p "$OUT"
"$CC" -O2 "${CONFIG[@]}" -I"$SRC" -c "$SRC/sqlite3-stock.c" -o "$OUT/sqlite3.o"
"$CC" -O2 -I"$SRC" "$SRC/shell.c" "$OUT/sqlite3.o" -lpthread -ldl -lm -o "$OUT/sqlite3"
"$CC" -O2 -I"$SRC" -DSQLITE_STUDY_ORACLE "$SRC/speedtest1.c" "$OUT/sqlite3.o" -lpthread -ldl -lm \
  -o "$OUT/speedtest1"
