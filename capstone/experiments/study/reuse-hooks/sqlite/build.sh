#!/usr/bin/env bash
# SQLite 3.22.0 speedtest1 with the reuse-distance hooks, one arm.
#   usage: build.sh <sdk dir> <prepared sources (prepare-sources.sh)> <out dir> <stock|protected>
# The memsys5 steps of ports/sqlite/app/build-domain.sh (same compiler, flags and
# lifetime patch), plus one step: hooks-<arm>.patch is applied to the engine copy
# and the engine is compiled with -DEXP_REUSE_GAP. The port script itself is not
# changed. Writes <out>/speedtest1.dom and <out>/src/sqlite3-capstone.c.
set -euo pipefail
SDK=${1:?usage: build.sh <sdk> <sources> <out> <stock|protected>}
SRC=${2:?usage}
OUT=${3:?usage}
ARM=${4:?usage}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../../.." && pwd)
CC="$SDK/capstone-cc"
[[ ! -e "$OUT" ]] || { echo "use a fresh output directory: $OUT" >&2; exit 2; }
mkdir -p "$OUT/src"
ENGINE=$OUT/src/sqlite3-capstone.c
cp "$SRC/sqlite3.c" "$ENGINE"
case "$ARM" in
  stock) ;;
  protected)
    patch -s -F0 -p1 -d "$OUT/src" < "$REPO/capstone/ports/sqlite/sublet/sublet-3220000-memsys5.patch" ;;
  *) echo "arm is stock or protected, not $ARM" >&2; exit 2 ;;
esac
patch -s -F0 -p1 -d "$OUT/src" < "$HERE/hooks-$ARM.patch"
cp "$HERE/exp-reuse-gap.h" "$REPO/capstone/experiments/study/reuse-gap-observer.h" "$OUT/src/"
CONFIG=(-DHAVE_MALLOC_H=1 -DHAVE_MALLOC_USABLE_SIZE=1 -DSQLITE_ENABLE_MEMSYS5
        -I"$REPO/capstone/runtime/include" -DEXP_REUSE_GAP)
"$CC" -O2 "${CONFIG[@]}" -I"$SRC" -c "$ENGINE" -o "$OUT/sqlite3.o"
"$CC" -O2 -I"$SRC" -DSQLITE_STUDY_ORACLE "$SRC/speedtest1.c" "$OUT/sqlite3.o" \
  -lpthread -ldl -lm -o "$OUT/speedtest1.dom"
sha256sum "$ENGINE" "$OUT/src/exp-reuse-gap.h" "$OUT/src/reuse-gap-observer.h" \
  "$OUT/speedtest1.dom"
