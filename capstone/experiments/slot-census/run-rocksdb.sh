#!/bin/bash
# Slot census of RocksDB's db_bench across key counts. RocksDB keeps its memtable entries in an arena of blocks it
# mallocs, so those entries are hidden from the allocator below; the block cache, table readers, iterators and
# write batches are separate mallocs.
#
# usage: run-rocksdb.sh OUT_DIR [KEYS...]   (default 1000000 10000000)
# Clones RocksDB $VERSION and gflags $GFLAGS (db_bench needs gflags; the measuring hosts have no package) from
# GitHub, builds gflags static and db_bench (Release, PORTABLE, no compression libraries, no jemalloc). One cell
# per key count: db_bench --benchmarks=fillrandom,readrandom,seekrandom,overwrite --num=<keys> with 100-byte values,
# a 1 GiB block cache, no compression and its own fresh database. A cell is valid when db_bench exits 0.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; shift
KEYS=("$@"); [ ${#KEYS[@]} -gt 0 ] || KEYS=(1000000 10000000)
VERSION=v10.10.1
GFLAGS=v2.2.2
BENCHMARKS=fillrandom,readrandom,seekrandom,overwrite
[ -e "$OUT" ] && { echo "run-rocksdb: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT/logs"; OUT=$(cd "$OUT" && pwd); TOOL=$OUT/tool
. "$HERE/tracer.sh"
build_tracer
run_controls

git -c advice.detachedHead=false clone -q --depth 1 -b "$GFLAGS" https://github.com/gflags/gflags "$TOOL/gflags-src" &&
git -c advice.detachedHead=false clone -q --depth 1 -b "$VERSION" https://github.com/facebook/rocksdb "$TOOL/rocksdb" \
  || exit 1
(cmake -S "$TOOL/gflags-src" -B "$TOOL/gflags-build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
   -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX="$TOOL/gflags" \
   -DCMAKE_POLICY_VERSION_MINIMUM=3.5 > "$OUT/logs/build.log" 2>&1 &&   # gflags 2.2.2 predates CMake 4
 cmake --build "$TOOL/gflags-build" -j16 --target install >> "$OUT/logs/build.log" 2>&1 &&
 cmake -S "$TOOL/rocksdb" -B "$TOOL/rocksdb-build" -DCMAKE_BUILD_TYPE=Release -DPORTABLE=1 -DWITH_GFLAGS=ON \
   -DCMAKE_PREFIX_PATH="$TOOL/gflags" -DWITH_TESTS=OFF -DWITH_BENCHMARK_TOOLS=ON -DWITH_TOOLS=OFF \
   -DWITH_JEMALLOC=OFF -DWITH_SNAPPY=OFF -DWITH_LZ4=OFF -DWITH_ZSTD=OFF -DWITH_ZLIB=OFF -DWITH_BZ2=OFF \
   -DFAIL_ON_WARNINGS=OFF >> "$OUT/logs/build.log" 2>&1 &&
 cmake --build "$TOOL/rocksdb-build" -j32 --target db_bench >> "$OUT/logs/build.log" 2>&1) \
  || { echo "run-rocksdb: build failed" >&2; exit 1; }
DB=$TOOL/rocksdb-build/db_bench
{
  provenance_tracer
  echo "rocksdb $VERSION $(git -C "$TOOL/rocksdb" rev-parse HEAD)"
  echo "gflags $GFLAGS $(git -C "$TOOL/gflags-src" rev-parse HEAD)"
  echo "build cmake Release PORTABLE=1, no compression, no jemalloc; target db_bench"
  echo "keys ${KEYS[*]}; db_bench --benchmarks=$BENCHMARKS --value_size=100 --cache_size=1073741824"
  echo "  --compression_type=none (db_bench defaults to snappy, which this build does not link)"
} > "$OUT/provenance.txt"

: > "$OUT/cells.tsv"
for keys in "${KEYS[@]}"; do
  cell=rocksdb-$keys
  rm -rf "$TOOL/db"
  t=$(date +%s)
  measured "$OUT/$cell.jsonl" \
    "$DB" --benchmarks="$BENCHMARKS" --num="$keys" --value_size=100 --cache_size=1073741824 --compression_type=none \
    --db="$TOOL/db" \
    > "$OUT/$cell.out" 2>&1
  rc=$?
  v=INVALID; [ $rc = 0 ] && v=valid
  printf '%s\tdb_bench\t%s\trc=%s %ss\n' "$cell" "$v" "$rc" "$(( $(date +%s) - t ))" | tee -a "$OUT/cells.tsv"
  rm -rf "$TOOL/db"
done
echo CENSUS_DONE >> "$OUT/cells.tsv"
