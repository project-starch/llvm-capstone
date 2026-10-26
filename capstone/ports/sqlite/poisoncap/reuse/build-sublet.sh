#!/usr/bin/env bash
# The Capstone side of the reuse-distance comparison: PoisonCap's speedtest1 driver on virtual Capstone,
# with the same wrapper, against three memsys5 arms, and the wrapper's test programs per arm.
#
#   bash build-sublet.sh SDK AMALGAMATION OUT
#     SDK           a virtual-profile SDK directory (capstone-cc, sdk.json)
#     AMALGAMATION  SQLite 3.22.0's amalgamation directory (sqlite3.c 7fc92e19..., sqlite3.h)
#
# Arms, each OUT/<arm>/ (ARMS overrides the list):
#   block-bounds   memsys5 on its Sublet port with memsys5's own power-of-two rounding
#                  (sublet-3220000-memsys5.patch + sublet-block-bounds.py): spatial + temporal; the
#                  primary arm, matched to PoisonCap's allocation policy and bounds granularity
#   spatial        the same without per-object revocation (+ sublet-spatial-only.py): its control
#   sublet         the port as ported (request-sized bounds and sizes): a side note
#   stock          memsys5 as SQLite ships it (no Sublet)
# The engine source is staged as build-virtual.py stages it (the amalgamation with the corpus's two FTS
# fixes, then the patches), compiled with the feature defines of PoisonCap's own build
# (OPT_FEATURE_FLAGS and TCC there: OMIT_LOAD_EXTENSION, ENABLE_MEMSYS5, ENABLE_COLUMN_METADATA,
# NDEBUG, THREADSAFE=1) at -O0, their build's level. The driver is make-speedtest1-capstone.py's
# copy of their speedtest1.c (one dead function removed) with shim/ for its CheriBSD headers.
set -euo pipefail
SDK=$(cd "$1" && pwd); AMALG=$(cd "$2" && pwd); OUT=$3
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PORT=$(cd "$HERE/../.." && pwd)            # ports/sqlite
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
THEIRS=${POISONCAP_SPEEDTEST1_C:-$HOME/poisoncap-platform/work/source/sqlite/test/speedtest1.c}
CC=$SDK/capstone-cc
DEFS=(-O0 -DHAVE_MALLOC_H=1 -DHAVE_MALLOC_USABLE_SIZE=1 -DSQLITE_OMIT_LOAD_EXTENSION=1 -DSQLITE_ENABLE_MEMSYS5
      -DSQLITE_ENABLE_COLUMN_METADATA=1 -DNDEBUG -DSQLITE_THREADSAFE=1)
NM=$(dirname "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cc"])' "$SDK/sdk.json")")/llvm-nm
WRAP=(-Wl,--wrap=sqlite3_config -Wl,--wrap=sqlite3_step -Wl,--wrap=sqlite3_reset)
[ "$(sha256sum "$AMALG/sqlite3.c" | cut -c1-16)" = 7fc92e19a3af61f3 ] || { echo "not the 3.22.0 amalgamation" >&2; exit 2; }
# The artifact's speedtest1.c (speedtest1/build.sh's source tree holds whichever variant was built last).
[ "$(sha256sum "$THEIRS" | cut -c1-16)" = 49d0815e03640f10 ] || { echo "$THEIRS is not the artifact's speedtest1.c" >&2; exit 2; }
mkdir -p "$OUT/common"
python3 "$HERE/make-speedtest1-capstone.py" "$THEIRS" "$OUT/common/speedtest1.c"
"$CC" "${DEFS[@]}" -I"$HERE/shim" -I"$AMALG" -c "$OUT/common/speedtest1.c" -o "$OUT/common/speedtest1.o"
"$CC" "${DEFS[@]}" -I"$AMALG" -c "$HERE/reuse-wrap.c" -o "$OUT/common/reuse-wrap.o"
"$CC" "${DEFS[@]}" -DREUSE_WRAP_TRACE -I"$AMALG" -c "$HERE/reuse-wrap.c" -o "$OUT/common/reuse-wrap-trace.o"
"$CC" "${DEFS[@]}" -I"$AMALG" -c "$HERE/test-reuse-wrap.c" -o "$OUT/common/test.o"
"$CC" "${DEFS[@]}" -c "$HERE/snapshot-stub.c" -o "$OUT/common/snapshot-stub.o"

for arm in ${ARMS:-block-bounds spatial sublet stock}; do
  A=$OUT/$arm; mkdir -p "$A/src"
  S=$A/src/sqlite3-capstone.c
  cp "$AMALG/sqlite3.c" "$S"
  python3 "$PORT/fts5-azarg-patch.py" "$S" > /dev/null
  python3 "$PORT/fts3-doclist-null-patch.py" "$S" > /dev/null
  INC=()
  if [ "$arm" != stock ]; then
    patch -s -F0 -p1 -d "$A/src" < "$PORT/sublet/sublet-3220000-memsys5.patch"
    INC=(-I"$REPO/capstone/runtime/include")
  fi
  case $arm in block-bounds|spatial) python3 "$PORT/sublet/sublet-block-bounds.py" "$S" > /dev/null ;; esac
  [ "$arm" != spatial ] || python3 "$PORT/sublet/sublet-spatial-only.py" "$S" > /dev/null
  "$CC" "${DEFS[@]}" "${INC[@]}" -I"$AMALG" -c "$S" -o "$A/sqlite3.o"
  "$CC" "$OUT/common/speedtest1.o" "$OUT/common/reuse-wrap.o" "$A/sqlite3.o" "${WRAP[@]}" -lm -o "$A/speedtest1.dom"
  "$CC" "$OUT/common/speedtest1.o" "$A/sqlite3.o" -lm -o "$A/speedtest1-plain.dom"
  "$CC" "$OUT/common/test.o" "$OUT/common/reuse-wrap.o" "$A/sqlite3.o" "${WRAP[@]}" -lm -o "$A/test-wrapped.dom"
  "$CC" "$OUT/common/test.o" "$OUT/common/reuse-wrap-trace.o" "$A/sqlite3.o" "${WRAP[@]}" -lm -o "$A/test-traced.dom"
  "$CC" "$OUT/common/test.o" "$OUT/common/snapshot-stub.o" "$A/sqlite3.o" -lm -o "$A/test-plain.dom"
  for d in speedtest1 test-wrapped; do
    # The symbols into a variable first: `nm | grep -q` fails under pipefail when grep stops early.
    syms=$("$NM" "$A/$d.dom")
    for s in __wrap_sqlite3_config __wrap_sqlite3_step __wrap_sqlite3_reset; do
      grep -q " T $s\$" <<< "$syms" || { echo "$arm/$d: $s not linked" >&2; exit 3; }
    done
  done
  echo "$arm: $(sha256sum "$A/speedtest1.dom" | cut -c1-16)"
done
sha256sum "$OUT"/*/*.dom "$OUT/common/speedtest1.c" "$HERE/reuse-wrap.c" > "$OUT/SHA256SUMS"
