#!/bin/bash
# Slot census of SQLite 3.22.0's speedtest1 (testset main) across dataset sizes, with SQLite's lookaside allocator
# on (its default, 1200 slots of 100 bytes per connection) and off. Lookaside serves small allocations from one
# malloc'd block, so the tracer sees them only with it off: the pair measures how many objects a nested allocator
# hides from the allocator below.
#
# usage: run-sqlite.sh OUT_DIR [SIZE...]   (default sizes 25 50 100 200 400 800; speedtest1's own default is 100)
# Downloads the 3.22.0 amalgamation and source archive from sqlite.org into $SLOTCENSUS_CACHE (default
# ~/slot-census/cache) and records their sha256. Each cell writes OUT_DIR/sqlite-<arm>-<size>.jsonl (census) and .out
# (speedtest1's output with --stats); a cell is valid when speedtest1 exits 0 and prints its TOTAL line.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; shift
SIZES=("$@"); [ ${#SIZES[@]} -gt 0 ] || SIZES=(25 50 100 200 400 800)
CACHE=${SLOTCENSUS_CACHE:-$HOME/slot-census/cache}
[ -e "$OUT" ] && { echo "run-sqlite: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT" "$CACHE"; OUT=$(cd "$OUT" && pwd); TOOL=$OUT/tool
. "$HERE/tracer.sh"
build_tracer
run_controls

# SQLite 3.22.0 and speedtest1 from the same release
for z in sqlite-amalgamation-3220000.zip sqlite-src-3220000.zip; do
  [ -s "$CACHE/$z" ] || curl -fsSL -o "$CACHE/$z" "https://www.sqlite.org/2018/$z" || exit 1
done
SRC=$TOOL/sqlite; mkdir -p "$SRC"
unzip -q -o -j "$CACHE/sqlite-amalgamation-3220000.zip" -d "$SRC" || exit 1
unzip -q -o -j "$CACHE/sqlite-src-3220000.zip" sqlite-src-3220000/test/speedtest1.c -d "$SRC" || exit 1
grep -q '#define SQLITE_VERSION *"3.22.0"' "$SRC/sqlite3.h" || { echo "run-sqlite: not 3.22.0" >&2; exit 1; }
gcc -O2 -I"$SRC" -o "$TOOL/speedtest1" "$SRC/speedtest1.c" "$SRC/sqlite3.c" -lpthread -ldl -lm || exit 1
{
  provenance_tracer
  (cd "$CACHE" && sha256sum sqlite-amalgamation-3220000.zip sqlite-src-3220000.zip)
  (cd "$TOOL" && sha256sum speedtest1)
  echo "build gcc -O2 speedtest1.c sqlite3.c (amalgamation defaults)"
  echo "sizes ${SIZES[*]}"
} > "$OUT/provenance.txt"

: > "$OUT/cells.tsv"
for size in "${SIZES[@]}"; do
  for arm in lookaside nolookaside; do
    cell=sqlite-$arm-$size
    look=(); [ $arm = nolookaside ] && look=(--lookaside 0 0)
    rm -f "$OUT/$cell.db"
    t=$(date +%s)
    SLOTCENSUS_OUT=$OUT/$cell.jsonl LD_PRELOAD=$TOOL/slotcensus.so \
      "$TOOL/speedtest1" --size "$size" --stats "${look[@]}" "$OUT/$cell.db" > "$OUT/$cell.out" 2>&1
    rc=$?
    v=INVALID; [ $rc = 0 ] && grep -q 'TOTAL' "$OUT/$cell.out" && v=valid
    printf '%s\tspeedtest1\t%s\trc=%s %ss\n' "$cell" "$v" "$rc" "$(( $(date +%s) - t ))" | tee -a "$OUT/cells.tsv"
    rm -f "$OUT/$cell.db"
  done
done
echo CENSUS_DONE >> "$OUT/cells.tsv"
