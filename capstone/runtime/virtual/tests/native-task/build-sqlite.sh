#!/usr/bin/env bash
# Sources are the existing prepared SQLite application port, kept outside git.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
src=${1:?external prepared SQLite source directory}
out=${2:?output directory}
: "${CAPSTONE_EXCHANGE_CC:?exchange SDK compiler}" "${CAPSTONE_NATIVE_CC:?direct IO SDK compiler}"
mkdir -p "$out"
"$CAPSTONE_EXCHANGE_CC" -O2 -DHAVE_MALLOC_H=1 -DHAVE_MALLOC_USABLE_SIZE=1 -I"$src" -c "$src/sqlite3.c" -o "$out/sqlite3.o"
"$CAPSTONE_EXCHANGE_CC" -O2 -I"$src" -DSQLITE_STUDY_ORACLE -c "$src/speedtest1.c" -o "$out/speedtest1.o"
"$CAPSTONE_EXCHANGE_CC" -O2 "$out/speedtest1.o" "$out/sqlite3.o" -lpthread -ldl -lm -o "$out/speedtest1-exchange.dom"
"$CAPSTONE_NATIVE_CC" -O2 "$out/speedtest1.o" "$out/sqlite3.o" -lpthread -ldl -lm -o "$out/speedtest1-direct.dom"
sha256sum "$src"/{sqlite3.c,sqlite3.h,sqlite3ext.h,speedtest1.c} "$out"/{sqlite3.o,speedtest1.o,speedtest1-exchange.dom,speedtest1-direct.dom} > "$out/sqlite-inputs.sha256"
