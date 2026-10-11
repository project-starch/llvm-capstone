#!/usr/bin/env bash
# SQLite 3.22.0 as an ordinary program for the delegated runtime: the unix VFS against musl,
# compiled and linked by the application SDK's capstone-cc, with SQLite's own defaults. Nothing
# of the freestanding build (../build-sqlite-capstone.sh: OS_OTHER, the libc shim, memsys5, no
# floating point) is used; the only source changes are the Capstone adaptations of
# ../adapt-sqlite-322.sh.
#   usage: build-domain.sh <sdk dir> <prepared sources (prepare-sources.sh)> <out dir>
# Writes <out>/sqlite3.dom (the shell) and <out>/speedtest1.dom (with the result oracle).
# SQLITE_OPT_LEVEL selects the optimisation level (default -O2).
set -euo pipefail
SDK=${1:?usage: build-domain.sh <sdk> <sources> <out>}
SRC=${2:?usage: build-domain.sh <sdk> <sources> <out>}
OUT=${3:?usage: build-domain.sh <sdk> <sources> <out>}
CC="$SDK/capstone-cc"
OPT=${SQLITE_OPT_LEVEL:--O2}
# What configure defines on Linux, and the one configuration this port depends on: with them
# SQLite asks the heap for a block's size (malloc_usable_size) instead of putting an 8-byte
# size header in front of every block, which leaves every structure it allocates that holds a
# capability 8 bytes off its 16-byte boundary (its first mutex faults, cause 6). build-native.sh
# uses the same two.
CONFIG=(-DHAVE_MALLOC_H=1 -DHAVE_MALLOC_USABLE_SIZE=1)
mkdir -p "$OUT"
"$CC" "$OPT" "${CONFIG[@]}" -I"$SRC" -c "$SRC/sqlite3.c" -o "$OUT/sqlite3.o"
"$CC" "$OPT" -I"$SRC" "$SRC/shell.c" "$OUT/sqlite3.o" -lpthread -ldl -lm -o "$OUT/sqlite3.dom"
"$CC" "$OPT" -I"$SRC" -DSQLITE_STUDY_ORACLE "$SRC/speedtest1.c" "$OUT/sqlite3.o" \
  -lpthread -ldl -lm -o "$OUT/speedtest1.dom"
sha256sum "$OUT/sqlite3.dom" "$OUT/speedtest1.dom"
