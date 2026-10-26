#!/bin/bash
# Slot census of a PostgreSQL server under pgbench across scale factors. PostgreSQL allocates through its own memory
# contexts (palloc), carved from blocks of 8 KiB and up that it mallocs, so the tracer sees blocks, not palloc
# chunks: the counts are the objects the allocator below would see if PostgreSQL ran unmodified.
#
# usage: run-postgres.sh OUT_DIR [SCALE...]   (default 1 10 100; scale 1 is 100,000 pgbench_accounts rows)
# Downloads postgresql-$VERSION from ftp.postgresql.org into $SLOTCENSUS_CACHE (default ~/slot-census/cache) and
# records its sha256. One cell per scale: initdb (untraced), start the server under the tracer (the postmaster and
# every backend it forks report), pgbench -i -s <scale>, then pgbench's TPC-B-like script with $CLIENTS clients for
# $SECONDS_RUN seconds, then pg_ctl stop -m fast. A cell's measured processes are every postgres process (matched
# by command line, which a backend rewrites to "postgres: ..."); a backend is a fork child and starts at the live
# set it inherits. A cell is valid when both pgbench runs exit 0, none of
# its transactions failed, and the server stopped cleanly.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; shift
SCALES=("$@"); [ ${#SCALES[@]} -gt 0 ] || SCALES=(1 10 100)
VERSION=17.6
CACHE=${SLOTCENSUS_CACHE:-$HOME/slot-census/cache}
PORT=${SLOTCENSUS_PG_PORT:-25432}
CLIENTS=${SLOTCENSUS_PG_CLIENTS:-16}
SECONDS_RUN=${SLOTCENSUS_PG_SECONDS:-120}
[ -e "$OUT" ] && { echo "run-postgres: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT/logs" "$CACHE"; OUT=$(cd "$OUT" && pwd); TOOL=$OUT/tool
. "$HERE/tracer.sh"
build_tracer
run_controls

T=postgresql-$VERSION.tar.bz2
[ -s "$CACHE/$T" ] || curl -fsSL -o "$CACHE/$T" "https://ftp.postgresql.org/pub/source/v$VERSION/$T" || exit 1
tar xjf "$CACHE/$T" -C "$TOOL" || exit 1
(cd "$TOOL/postgresql-$VERSION" &&
 ./configure --prefix="$TOOL/pg" --without-icu --without-readline > "$OUT/logs/build.log" 2>&1 &&
 make -j32 >> "$OUT/logs/build.log" 2>&1 && make install >> "$OUT/logs/build.log" 2>&1) \
  || { echo "run-postgres: build failed" >&2; exit 1; }
B=$TOOL/pg/bin
{
  provenance_tracer
  (cd "$CACHE" && sha256sum "$T")
  echo "postgres $("$B/postgres" --version)"
  echo "build ./configure --without-icu --without-readline && make && make install"
  echo "scales ${SCALES[*]}; pgbench -i -s <scale>, then pgbench -c $CLIENTS -j $CLIENTS -T $SECONDS_RUN (TPC-B-like)"
} > "$OUT/provenance.txt"

: > "$OUT/cells.tsv"
for scale in "${SCALES[@]}"; do
  cell=postgres-$scale
  D=$TOOL/data-$scale
  t=$(date +%s)
  "$B/initdb" -D "$D" -A trust -U bench > "$OUT/logs/$cell.initdb" 2>&1 || exit 1
  SLOTCENSUS_OUT=$OUT/$cell.jsonl LD_PRELOAD=$TOOL/slotcensus.so \
    "$B/pg_ctl" -D "$D" -w -l "$OUT/logs/$cell.server" -o "-p $PORT -k $D -c listen_addresses=''" start \
    > "$OUT/logs/$cell.pgctl" 2>&1 || { echo "run-postgres: server did not start" >&2; exit 1; }
  {
    "$B/pgbench" -h "$D" -p "$PORT" -U bench -i -s "$scale" postgres 2>&1; echo "init rc=$?"
    "$B/pgbench" -h "$D" -p "$PORT" -U bench -c "$CLIENTS" -j "$CLIENTS" -T "$SECONDS_RUN" postgres 2>&1
    echo "run rc=$?"
  } > "$OUT/$cell.out"
  "$B/pg_ctl" -D "$D" -w -m fast stop >> "$OUT/logs/$cell.pgctl" 2>&1; stop=$?
  v=INVALID
  grep -q '^init rc=0' "$OUT/$cell.out" && grep -q '^run rc=0' "$OUT/$cell.out" &&
    grep -q 'number of failed transactions: 0' "$OUT/$cell.out" && [ $stop = 0 ] && v=valid
  printf '%s\tre:^(\\S*/)?postgres(:|\\s|$)\t%s\tstop=%s %ss\n' "$cell" "$v" "$stop" \
    "$(( $(date +%s) - t ))" | tee -a "$OUT/cells.tsv"
  rm -rf "$D"
done
echo CENSUS_DONE >> "$OUT/cells.tsv"
